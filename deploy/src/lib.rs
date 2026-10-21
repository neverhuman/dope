//! Development runtime. Native Rust owns the sampler and all support semantics.
//! SafeTensors is a transport, never evidence of compression or privacy quality.
use std::collections::{BTreeMap, HashMap};
use std::io::Read;
use std::path::Path;

use candle_core::{Device, Tensor};
use dope_kernel::codec::LoadedKernel;
use dope_kernel::model::{JointDecoder, JointNetwork, Kernel, KernelProgram, QuantizedLinear};
use dope_kernel::neural::deployment::{finish_tvae, tvae_latent};
use dope_kernel::sample::{SampleOptions, joint_row_seeds, sample_joint_ranks, sample_kernel};
use safetensors::tensor::{Dtype, SafeTensors, TensorView};

pub type Result<T> = std::result::Result<T, &'static str>;
pub const MAX_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_CELLS: usize = 1_000_000;
const FORMAT: &str = "dope-safetensors-carrier-v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Activation {
    LegacyV1,
    TanhV2,
}

impl Activation {
    fn name(self) -> &'static str {
        match self {
            Self::LegacyV1 => "legacy-v1",
            Self::TanhV2 => "tanh-v2",
        }
    }
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum DeviceRequest {
    Auto,
    Cpu,
    Cuda,
}

pub fn read_bounded(path: &Path) -> Result<Vec<u8>> {
    let mut opts = std::fs::OpenOptions::new();
    opts.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // Avoid following a final symlink or blocking on a FIFO.
        opts.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let file = opts.open(path).map_err(|_| "input_unavailable")?;
    let meta = file.metadata().map_err(|_| "input_unavailable")?;
    if !meta.is_file() || meta.len() > MAX_BYTES as u64 {
        return Err("input_budget_or_type");
    }
    let mut data = Vec::new();
    file.take(MAX_BYTES as u64 + 1)
        .read_to_end(&mut data)
        .map_err(|_| "input_unavailable")?;
    if data.len() > MAX_BYTES {
        return Err("input_budget_or_type");
    }
    Ok(data)
}

fn decoder(kernel: &Kernel) -> Option<&JointDecoder> {
    match &kernel.program {
        KernelProgram::NeuralJoint(g) => match &g.network {
            JointNetwork::Tvae { decoder } => Some(decoder),
            _ => None,
        },
        _ => None,
    }
}

type OwnedTensor = (Dtype, Vec<usize>, Vec<u8>);

fn tensor_inventory(native: &[u8], kernel: &Kernel) -> BTreeMap<String, OwnedTensor> {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "native_dpk".into(),
        (Dtype::U8, vec![native.len()], native.to_vec()),
    );
    if let Some(d) = decoder(kernel) {
        for (name, layer) in [
            ("hidden_1", &d.hidden_1),
            ("hidden_2", &d.hidden_2),
            ("value_head", &d.value_head),
            ("missing_head", &d.missing_head),
        ] {
            tensors.insert(
                format!("{name}.weight"),
                (
                    Dtype::I8,
                    vec![layer.output_dim as usize, layer.input_dim as usize],
                    layer.weights.iter().map(|v| *v as u8).collect(),
                ),
            );
            for (suffix, values) in [("scale", &layer.scales), ("bias", &layer.biases)] {
                tensors.insert(
                    format!("{name}.{suffix}"),
                    (
                        Dtype::F32,
                        vec![values.len()],
                        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
                    ),
                );
            }
        }
    }
    tensors
}

/// Version one retains DPK plus typed decoder tensors. Charge the entire file,
/// including duplicate parameters and the SafeTensors header. Never substitute
/// the smaller embedded DPK size in a tensor-runtime benchmark.
pub fn pack(native: &[u8], activation: Activation) -> Result<Vec<u8>> {
    if native.len() > MAX_BYTES / 2 {
        return Err("artifact_budget");
    }
    let kernel = dope_kernel::decode_kernel(native).map_err(|_| "native_artifact_invalid")?;
    if activation == Activation::TanhV2 && decoder(&kernel).is_none() {
        return Err("activation_requires_tvae");
    }
    let owned = tensor_inventory(native, &kernel);
    let views = owned
        .iter()
        .map(|(name, (dtype, shape, data))| {
            TensorView::new(*dtype, shape.clone(), data)
                .map(|view| (name.as_str(), view))
                .map_err(|_| "tensor_invalid")
        })
        .collect::<Result<Vec<_>>>()?;
    let metadata = HashMap::from([
        ("format".into(), FORMAT.into()),
        ("activation".into(), activation.name().into()),
    ]);
    let bytes =
        safetensors::tensor::serialize(views, Some(metadata)).map_err(|_| "tensor_invalid")?;
    if bytes.len() > MAX_BYTES {
        return Err("artifact_budget");
    }
    Ok(bytes)
}

pub struct Carrier {
    pub kernel: Kernel,
    pub activation: Activation,
    pub charged_bytes: usize,
}

impl Carrier {
    pub fn load(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > MAX_BYTES {
            return Err("artifact_budget");
        }
        let (_, header) = SafeTensors::read_metadata(bytes).map_err(|_| "tensor_invalid")?;
        let meta = header
            .metadata()
            .as_ref()
            .ok_or("tensor_metadata_invalid")?;
        if meta.len() != 2 || meta.get("format").map(String::as_str) != Some(FORMAT) {
            return Err("tensor_metadata_invalid");
        }
        let activation = match meta.get("activation").map(String::as_str) {
            Some("legacy-v1") => Activation::LegacyV1,
            Some("tanh-v2") => Activation::TanhV2,
            _ => return Err("tensor_activation_invalid"),
        };
        let tensors = SafeTensors::deserialize(bytes).map_err(|_| "tensor_invalid")?;
        let native = tensors
            .tensor("native_dpk")
            .map_err(|_| "native_tensor_missing")?;
        if native.dtype() != Dtype::U8 || native.shape() != [native.data().len()] {
            return Err("native_tensor_invalid");
        }
        let kernel =
            dope_kernel::decode_kernel(native.data()).map_err(|_| "native_artifact_invalid")?;
        if activation == Activation::TanhV2 && decoder(&kernel).is_none() {
            return Err("activation_requires_tvae");
        }
        // Authenticate every tensor against the canonical native graph. This
        // prevents an uncharged, unused or inconsistent alternate weight set.
        let expected = tensor_inventory(native.data(), &kernel);
        if tensors.names().len() != expected.len() {
            return Err("tensor_inventory_mismatch");
        }
        for (name, (dtype, shape, data)) in expected {
            let actual = tensors
                .tensor(&name)
                .map_err(|_| "tensor_inventory_mismatch")?;
            if actual.dtype() != dtype || actual.shape() != shape || actual.data() != data {
                return Err("tensor_inventory_mismatch");
            }
        }
        Ok(Self {
            kernel,
            activation,
            charged_bytes: bytes.len(),
        })
    }

    pub fn sample(
        &self,
        options: SampleOptions,
        requested: DeviceRequest,
    ) -> Result<(Vec<f32>, &'static str)> {
        let width = self.kernel.features as usize + 1;
        if options.rows == 0
            || options.rows > 100_000
            || options
                .rows
                .checked_mul(width)
                .is_none_or(|n| n > MAX_CELLS)
        {
            return Err("sampling_budget");
        }
        let Some(d) = decoder(&self.kernel) else {
            if matches!(requested, DeviceRequest::Cuda) {
                return Err("cuda_graph_unsupported");
            }
            let values = sample_kernel(&LoadedKernel::V3(Box::new(self.kernel.clone())), options)
                .map_err(|_| "native_sampling_failed")?;
            return Ok((values, "cpu-native"));
        };
        let (device, label) = select_device(requested)?;
        // The historical CPU stream remains bit exact. Corrected activation and
        // CUDA are explicit successor numerical profiles, tested by tolerance.
        if device.is_cpu() && self.activation == Activation::LegacyV1 {
            return sample_kernel(&LoadedKernel::V3(Box::new(self.kernel.clone())), options)
                .map(|v| (v, "cpu-native"))
                .map_err(|_| "native_sampling_failed");
        }
        let KernelProgram::NeuralJoint(g) = &self.kernel.program else {
            unreachable!()
        };
        let layers = TensorDecoder::new(d, &device)?;
        let seeds = joint_row_seeds(&self.kernel, options).map_err(|_| "sampling_seed_invalid")?;
        let latent_width = d.hidden_1.input_dim as usize;
        let mut output = Vec::with_capacity(options.rows * width);
        for batch in seeds.chunks(256) {
            let mut latents = Vec::with_capacity(batch.len() * latent_width);
            for seed in batch {
                latents.extend(tvae_latent(g, width - 1, *seed).map_err(|_| "latent_invalid")?);
            }
            let input = Tensor::from_vec(latents, (batch.len(), latent_width), &device)
                .map_err(|_| "tensor_execution_failed")?;
            let (parameters, logits) = layers.forward(&input, self.activation)?;
            let parameters = parameters
                .to_vec2::<f32>()
                .map_err(|_| "tensor_execution_failed")?;
            let logits = logits
                .to_vec2::<f32>()
                .map_err(|_| "tensor_execution_failed")?;
            let draws = batch
                .iter()
                .zip(parameters)
                .zip(logits)
                .map(|((seed, p), m)| {
                    finish_tvae(g, width - 1, *seed, &p, &m).map_err(|_| "decoder_output_invalid")
                })
                .collect::<Result<Vec<_>>>()?;
            output.extend(
                sample_joint_ranks(&self.kernel, &draws).map_err(|_| "native_support_failed")?,
            );
        }
        Ok((output, label))
    }
}

fn select_device(requested: DeviceRequest) -> Result<(Device, &'static str)> {
    if matches!(requested, DeviceRequest::Cpu) {
        return Ok((Device::Cpu, "cpu-candle"));
    }
    #[cfg(feature = "cuda")]
    if let Ok(device) = Device::new_cuda(0) {
        return Ok((device, "cuda-candle"));
    }
    if matches!(requested, DeviceRequest::Cuda) {
        return Err("cuda_unavailable");
    }
    Ok((Device::Cpu, "cpu-candle"))
}

struct Linear {
    weight: Tensor,
    scale: Tensor,
    bias: Tensor,
}

impl Linear {
    fn new(layer: &QuantizedLinear, device: &Device) -> Result<Self> {
        let weight = Tensor::from_vec(
            layer
                .weights
                .iter()
                .map(|v| f32::from(*v))
                .collect::<Vec<_>>(),
            (layer.output_dim as usize, layer.input_dim as usize),
            device,
        )
        .map_err(|_| "tensor_allocation_failed")?;
        let scale = Tensor::from_vec(layer.scales.clone(), layer.scales.len(), device)
            .map_err(|_| "tensor_allocation_failed")?;
        let bias = Tensor::from_vec(layer.biases.clone(), layer.biases.len(), device)
            .map_err(|_| "tensor_allocation_failed")?;
        Ok(Self {
            weight,
            scale,
            bias,
        })
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input
            .matmul(&self.weight.t().map_err(|_| "tensor_execution_failed")?)
            .and_then(|x| x.broadcast_mul(&self.scale))
            .and_then(|x| x.broadcast_add(&self.bias))
            .map_err(|_| "tensor_execution_failed")
    }
}

struct TensorDecoder {
    first: Linear,
    second: Linear,
    value: Linear,
    missing: Linear,
}

impl TensorDecoder {
    fn new(d: &JointDecoder, device: &Device) -> Result<Self> {
        Ok(Self {
            first: Linear::new(&d.hidden_1, device)?,
            second: Linear::new(&d.hidden_2, device)?,
            value: Linear::new(&d.value_head, device)?,
            missing: Linear::new(&d.missing_head, device)?,
        })
    }
    fn forward(&self, input: &Tensor, activation: Activation) -> Result<(Tensor, Tensor)> {
        fn gelu(x: Tensor, activation: Activation) -> Result<Tensor> {
            let coefficient = match activation {
                Activation::LegacyV1 => std::f64::consts::FRAC_2_SQRT_PI,
                Activation::TanhV2 => std::f64::consts::FRAC_2_PI.sqrt(),
            };
            let result = (|| -> candle_core::Result<Tensor> {
                let cube = x.sqr()?.mul(&x)?.affine(0.044_715, 0.0)?;
                let gate = x
                    .add(&cube)?
                    .affine(coefficient, 0.0)?
                    .tanh()?
                    .affine(1.0, 1.0)?;
                x.mul(&gate)?.affine(0.5, 0.0)
            })();
            result.map_err(|_| "tensor_execution_failed")
        }
        let hidden = gelu(self.first.forward(input)?, activation)?;
        let hidden = gelu(self.second.forward(&hidden)?, activation)?;
        Ok((self.value.forward(&hidden)?, self.missing.forward(&hidden)?))
    }
}

#[cfg(test)]
mod tests;
