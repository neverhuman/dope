//! GET-only GitHub API and bounded pagination, with injectable transport.
use super::{Refused, Result, array, integer, require};
use serde_json::{Value, json};

pub mod process;
pub use process::ProcessTransport;

pub struct Response {
    pub code: i32,
    pub stdout: String,
    pub stderr: String,
}
pub trait Transport {
    fn run(&mut self, argv: &[String]) -> Result<Response>;
}
pub trait Api {
    fn get(&mut self, endpoint: &str, optional: bool, paginate: bool) -> Result<Value>;
    fn calls(&self) -> Value;
    fn listing(&mut self, endpoint: &str, key: Option<&str>) -> Result<Value> {
        let mut items = Vec::new();
        let mut total = None;
        for page in 1..=10 {
            let separator = if endpoint.contains('?') { '&' } else { '?' };
            let data = self.get(
                &format!("{endpoint}{separator}per_page=100&page={page}"),
                false,
                false,
            )?;
            let batch = if let Some(key) = key {
                require(
                    data.is_object() && data[key].is_array() && data.get("total_count").is_some(),
                    "api_shape_invalid",
                )?;
                let count =
                    integer(&data["total_count"]).map_err(|_| Refused("api_shape_invalid"))?;
                require(total.is_none_or(|t| t == count), "api_listing_changed")?;
                total = Some(count);
                require(count < 1000, "api_listing_truncated")?;
                array(&data[key])?
            } else {
                array(&data).map_err(|_| Refused("api_shape_invalid"))?
            };
            require(batch.len() <= 100, "api_shape_invalid")?;
            items.extend(batch.iter().cloned());
            if batch.len() < 100 {
                require(
                    total.is_none_or(|n| n == items.len() as u64),
                    "api_listing_truncated",
                )?;
                return Ok(Value::Array(items));
            }
        }
        Err(Refused("api_listing_truncated"))
    }
    fn open_pulls(&mut self, root: &str) -> Result<Value> {
        let data = self.get(
            &format!("{root}/pulls?state=open&sort=created&direction=asc&per_page=100"),
            false,
            true,
        )?;
        let pages = array(&data).map_err(|_| Refused("open_pr_listing_unavailable"))?;
        require(!pages.is_empty(), "open_pr_listing_unavailable")?;
        require(pages.len() < 10, "open_pr_listing_truncated")?;
        let mut pulls = Vec::new();
        for (i, page) in pages.iter().enumerate() {
            let batch = array(page).map_err(|_| Refused("open_pr_listing_unavailable"))?;
            require(
                batch.len() <= 100 && (i == pages.len() - 1 || batch.len() == 100),
                "open_pr_listing_truncated",
            )?;
            pulls.extend(batch.iter().cloned());
        }
        Ok(Value::Array(pulls))
    }
}
pub struct GitHub<T> {
    pub executable: String,
    pub transport: T,
    calls: Vec<Value>,
}
impl<T: Transport> GitHub<T> {
    pub fn new(executable: String, transport: T) -> Self {
        Self {
            executable,
            transport,
            calls: Vec::new(),
        }
    }
}
impl<T: Transport> Api for GitHub<T> {
    fn calls(&self) -> Value {
        json!(self.calls)
    }
    fn get(&mut self, endpoint: &str, optional: bool, paginate: bool) -> Result<Value> {
        let mut argv = vec![
            self.executable.clone(),
            "api".into(),
            "--method".into(),
            "GET".into(),
            endpoint.into(),
        ];
        if paginate {
            argv.extend(["--paginate".into(), "--slurp".into()]);
        }
        let mut call = json!({"endpoint":endpoint,"method":"GET","state":"started"});
        if paginate {
            call["pagination"] = json!("all_linked_pages");
        }
        self.calls.push(call);
        let index = self.calls.len() - 1;
        let response = self.transport.run(&argv);
        let parsed = response.and_then(|r| {
            // Cargo diagnostics stay internal; GitHub receipts discard raw stderr.
            drop(r.stderr);
            let data: Value =
                serde_json::from_str(&r.stdout).map_err(|_| Refused("api_unavailable"))?;
            Ok((r.code, data))
        });
        let (code, data) = match parsed {
            Ok(v) => v,
            Err(_) => {
                self.calls[index]["state"] = json!("api_unavailable");
                return Err(Refused("api_unavailable"));
            }
        };
        self.calls[index]["state"] = json!("completed");
        self.calls[index]["exit_code"] = json!(code);
        self.calls[index]["sha256"] = json!(super::json::digest(&data)?);
        if code != 0 {
            if optional
                && data.is_object()
                && (data["status"] == "404" || data["status"].as_u64() == Some(404))
            {
                return Ok(Value::Null);
            }
            return Err(Refused("api_unavailable"));
        }
        Ok(data)
    }
}
pub fn quote(value: &str) -> String {
    value
        .bytes()
        .map(|b| {
            if b.is_ascii_alphanumeric() || b"_.-~".contains(&b) {
                (b as char).to_string()
            } else {
                format!("%{b:02X}")
            }
        })
        .collect()
}
