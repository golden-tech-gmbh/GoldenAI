use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::SupportedModels;
use crate::message::Message;

#[derive(Serialize, Clone, Debug)]
#[pyclass(dict, get_all, set_all, subclass)]
pub struct AnthropicRequest {
    pub(crate) model: SupportedModels,
    pub(crate) system: Option<String>,
    pub(crate) max_tokens: u32,
    pub(crate) messages: Vec<Message>,
}

#[pymethods]
impl AnthropicRequest {
    #[new]
    #[pyo3(signature = (model, max_tokens,messages,prompt=None))]
    fn new(
        model: &str,
        max_tokens: u32,
        messages: Vec<Message>,
        prompt: Option<&str>,
    ) -> PyResult<Self> {
        Ok(Self {
            model: SupportedModels::from_str(model).unwrap(),
            max_tokens,
            messages,
            system: prompt.map(|s| s.to_string()),
        })
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{self:?}"))
    }
}

#[derive(Deserialize, Debug, Clone)]
#[pyclass(dict, get_all, set_all)]
pub struct ResponseAnthropic {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: String,
}

#[pymethods]
impl ResponseAnthropic {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("ResponseAnthropic<text={:?}>", self.text))
    }
}
