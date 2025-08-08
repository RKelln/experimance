# WORK ORDER – Image Server Updates for Feed the Fires
_Lead dev_: ⟨name⟩   _Due_: ⟨date⟩

> Goal: accept explicit `width/height`, respect
>   `workers=N`, and stay backward-compatible with Experimance.

---

## 1. Schema patch

1. **Update** `experimance_common.schemas_base.RenderRequest`  
   * add required `width: int`, `height: int`.  
   * bump `__version__` minor.

2. **Regenerate** Pydantic stubs as needed

---

## 2. Worker concurrency

* In `image_server/backend/runner.py` read `[backend].workers` from TOML.  
* Spawn that many async tasks, each calling `Backend.generate()`.  
* Change queue size to config `[backend].queue_size`.  
* Add `--workers` CLI override flag (optional).

---

## 3. Unit tests

* `tests/test_render_request_dims.py` – reject missing width/height.  
* `tests/test_worker_parallel.py` – mock backend sleep(0.1); assert ≥2 images/sec with `workers=6`.

---

## 4. Docs

* Update `services/image_server/README.md`  
  * new config keys and example for cloud (`workers=6`) vs local (`workers=1`).  
* Add change-log entry (v 0.3.0).

---

## 5. PR checklist

- [ ] unit tests pass  
- [ ] `make mypy` clean  
- [ ] backward-compat with Experimance (runs existing demo)  
