use pyo3::prelude::*;
use pyo3::pyclass;
use serde::{Deserialize, Deserializer};
use std::collections::HashMap;

#[derive(Deserialize, Debug, Clone)]
#[pyclass(dict, get_all, set_all)]
struct ResponseContent {
    #[serde(rename = "type")]
    content_type: String,
    text: String,
}

#[pymethods]
impl ResponseContent {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("ResponseContent<text={:?}>", self.text))
    }
}

#[derive(Deserialize, Debug)]
#[pyclass(dict, get_all, frozen)]
pub(crate) struct AnthropicResponse {
    id: String,
    #[serde(rename = "type")]
    response_type: String,
    role: String,
    content: Vec<ResponseContent>,
    model: String,
    stop_reason: Option<String>,
    #[serde(deserialize_with = "deserialize_filtered_map")]
    usage: HashMap<String, u32>,
}

fn deserialize_filtered_map<'de, D>(deserializer: D) -> Result<HashMap<String, u32>, D::Error>
where
    D: Deserializer<'de>,
{
    use serde_json::Value;

    let value: HashMap<String, Value> = HashMap::deserialize(deserializer)?;
    let mut result = HashMap::new();

    for (key, val) in value {
        match val.as_u64() {
            Some(n) => {
                let num = n.try_into().ok();
                if let Some(num) = num {
                    result.insert(key, num);
                }
            }
            None => {}
        }
    }

    Ok(result)
}

#[pymethods]
impl AnthropicResponse {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "AnthropicResponse<id={:?}, model={:?}>",
            self.id, self.model
        ))
    }

    fn __str__(&self) -> PyResult<String> {
        let content: String = self.content.iter().map(|c| c.text.clone()).collect();
        Ok(format!("{}", content))
    }
}
