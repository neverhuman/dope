"""Exact-teacher, predictor, and coreset controls. No source rows are stored here."""

from types import MappingProxyType

from research.transfer.evidence import canonical_json, digest


CONTROLS = frozenset({
    "real_n", "real_bootstrap_4n", "shuffled_label", "no_noise",
    "prequantization", "postquantization", "direct_predictor", "coreset", "distillation",
})


class ControlError(ValueError):
    """Stable codes only. Messages do not carry draw values."""


def _fail(code):
    raise ControlError(code)


def _pairs(draws, noises):
    _fail("invalid_draws") if len(draws) != len(noises) else None
    return tuple(zip(draws, noises))


def _rows(draws, noises):
    _fail("invalid_draws") if type(draws) is not tuple or not draws else None
    _fail("invalid_draws") if len(draws) != len(noises) else None
    rows = []
    for index, (x, noise) in enumerate(zip(draws, noises)):
        _fail("invalid_draws") if type(x) is not int or type(noise) is not int else None
        rows.append({"row_id": index, "x": x, "noise": noise})
    return tuple(rows)


class Teacher:
    """Frozen target and noise parameters. The form is part of the target hash."""

    __slots__ = (
        "_form", "_parameters", "_noise_scale", "precision",
        "parameter_sha256", "noise_sha256", "_frozen",
    )
    _FROZEN = frozenset({
        "_form", "_parameters", "_noise_scale", "precision",
        "parameter_sha256", "noise_sha256", "_frozen",
    })

    def __setattr__(self, name, value):
        if name in self._FROZEN and getattr(self, "_frozen", False):
            raise AttributeError("frozen_teacher")
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self._FROZEN:
            raise AttributeError("frozen_teacher")
        object.__delattr__(self, name)

    def __init__(self, form, parameters, noise_scale, precision):
        _fail("invalid_target_form") if form not in ("linear", "nonlinear") else None
        _fail("invalid_precision") if precision != "float64" else None
        _fail("invalid_parameters") if set(parameters) != {"quadratic", "linear", "bias"} else None
        _fail("invalid_parameters") if any(type(parameters[key]) is not int for key in parameters) else None
        _fail("invalid_noise") if type(noise_scale) is not int else None
        self._form = form
        self._parameters = MappingProxyType(dict(parameters))
        self._noise_scale = noise_scale
        self.precision = precision
        self.parameter_sha256 = digest({
            "form": form, "precision": precision, "parameters": dict(self._parameters),
        })
        self.noise_sha256 = digest({"noise_scale": noise_scale, "precision": precision})
        self._frozen = True

    @property
    def form(self):
        return self._form

    @property
    def parameters(self):
        return self._parameters

    @property
    def noise_scale(self):
        return self._noise_scale

    def value(self, x, noise):
        quadratic, linear, bias = (self.parameters["quadratic"], self.parameters["linear"], self.parameters["bias"])
        base = quadratic * x * x + linear * x + bias if self.form == "nonlinear" else linear * x + bias
        return base + self.noise_scale * noise


class MatchedTeacherFactorial:
    """One frozen teacher, support, and budget, crossed with the declared controls."""

    def __init__(self, teacher, search_budget, byte_budget, declared):
        _fail("invalid_budget") if type(search_budget) is not int or search_budget < 1 else None
        _fail("invalid_budget") if type(byte_budget) is not int or byte_budget < 1 else None
        _fail("invalid_factorial") if set(declared) != CONTROLS else None
        self.teacher = teacher
        self.search_budget = search_budget
        self.byte_budget = byte_budget
        self.declared = frozenset(declared)

    def teacher_for(self, control, form):
        _fail("undeclared_control") if control not in self.declared else None
        noise = 0 if control == "no_noise" else self.teacher.noise_scale
        return Teacher(form, self.teacher.parameters, noise, self.teacher.precision)

    def cell(self, target_form, x_mode, control, draws, noises):
        _fail("invalid_x_mode") if x_mode not in ("independent", "learned") else None
        teacher = self.teacher_for(control, target_form)
        rows = _rows(draws, noises)
        values = tuple(teacher.value(row["x"], row["noise"]) for row in rows)
        if control == "shuffled_label":
            label_values = values[-1:] + values[:-1]
        elif control == "postquantization":
            label_values = tuple((value // 10) * 10 for value in values)
        else:
            label_values = values
        stored = rows
        if control == "real_bootstrap_4n":
            stored = tuple({"base": row["row_id"], "rep": rep, "x": row["x"]} for rep in range(4) for row in rows)
        return {
            "target_form": teacher.form,
            "x_mode": x_mode,
            "control": control,
            "parameter_sha256": teacher.parameter_sha256,
            "noise_sha256": teacher.noise_sha256,
            "support_sha256": digest({"x": [row["x"] for row in rows]}),
            "draw_sha256": digest({"rows": [{"row_id": row["row_id"], "x": row["x"]} for row in rows]}),
            "label_sha256": digest({"labels": list(label_values)}),
            "value_sha256": digest({"values": list(label_values)}),
            "base_rows": len(rows),
            "stored_rows": len(stored),
            "search_budget": self.search_budget,
            "byte_budget": self.byte_budget,
            "precision": teacher.precision,
            "retained_real_rows": control in ("real_n", "real_bootstrap_4n", "shuffled_label", "coreset"),
        }


class DirectPredictorControl:
    """The stored teacher, charged as predictor bytes, with the same real-X outputs."""

    def __init__(self, teacher, draws, noises):
        self.teacher = teacher
        self.outputs = tuple(teacher.value(x, noise) for x, noise in _pairs(draws, noises))
        stored = {
            "form": teacher.form,
            "precision": teacher.precision,
            "parameters": dict(teacher.parameters),
            "noise_scale": teacher.noise_scale,
        }
        self.charged_bytes = len(canonical_json(stored))
        self.retained_real_rows = False
        self.output_sha256 = digest({"outputs": list(self.outputs)})

    def empirical_diagnostic(self, draws, noises):
        values = tuple(self.teacher.value(x, noise) for x, noise in _pairs(draws, noises))
        return digest({"outputs": list(values)})


def _coreset_export(rows):
    return canonical_json({"rows": [
        {"row_id": row["row_id"], "x": row["x"], "noise": row["noise"]} for row in rows
    ]})


class ByteMatchedCoreset:
    """Keep the longest real-row prefix whose serialized export fits the budget."""

    def __init__(self, draws, noises, byte_budget):
        rows = _rows(draws, noises)
        _fail("invalid_budget") if type(byte_budget) is not int or byte_budget < 1 else None
        kept = ()
        charged = 0
        for count in range(1, len(rows) + 1):
            export = _coreset_export(rows[:count])
            if len(export) > byte_budget:
                break
            kept = rows[:count]
            charged = len(export)
        _fail("invalid_budget") if not kept else None
        self.rows = kept
        self.charged_bytes = charged
        self.byte_budget = byte_budget
        self.retained_real_rows = True
        self.adaptations = ()
