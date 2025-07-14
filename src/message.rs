use crate::SupportedModels;
use base64::Engine;
use pyo3::exceptions::{PyException, PyTypeError};
use pyo3::prelude::PyAnyMethods;
use pyo3::types::PyType;
use pyo3::{Bound, FromPyObject, IntoPyObject, PyAny, PyResult, pyclass, pymethods};
use serde::{Serialize, Serializer};
use std::path::PathBuf;

#[derive(Serialize, Clone, Debug)]
#[pyclass(dict, get_all, set_all, subclass)]
pub struct DocumentSourceContent {
    // Anthropic schema
    #[serde(rename = "type")]
    pub(crate) content_type: String, // "base64"
    pub(crate) media_type: String, // "image/jpeg"
    pub(crate) data: String,       // base64
}

#[pymethods]
impl DocumentSourceContent {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{self:?}"))
    }
}

#[derive(Serialize, Clone, Debug)]
#[pyclass(dict, get_all, set_all, subclass)]
pub struct DocumentContent {
    // Anthropic schema
    #[serde(rename = "type")]
    pub content_type: String, // "image" (Anthropic) or "input_file" (OpenAI)

    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<DocumentSourceContent>, // Anthropic

    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_data: Option<String>, // OpenAI

    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>, // OpenAI
}

#[pymethods]
impl DocumentContent {
    #[new]
    #[pyo3(signature = (path, llm=None))]
    pub fn new(path: &str, llm: Option<SupportedModels>) -> PyResult<Self> {
        let path = PathBuf::from(path);

        // Get file extension and convert to lowercase for case-insensitive matching
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_lowercase())
            .unwrap();

        // Determine content type and media type based on file extension
        let (content_type, media_type) = match ext.as_str() {
            // Image types
            "jpg" | "image" => ("image", "image/jpeg"),
            "png" => ("image", "image/png"),
            "gif" => ("image", "image/gif"),
            "bmp" => ("image", "image/bmp"),
            "webp" => ("image", "image/webp"),
            "svg" => ("image", "image/svg+xml"),
            "tiff" | "tif" => ("image", "image/tiff"),
            "heic" | "heif" => ("image", "image/heic"),

            // PDF
            "pdf" => ("document", "application/pdf"),

            // Unsupported type
            _ => {
                return Err(PyTypeError::new_err(format!(
                    "Unsupported file type: .{}",
                    ext
                )));
            }
        };

        let file_name = path
            .file_name()
            .and_then(|s| s.to_str())
            .map(|s| s.to_string())
            .unwrap();

        // Read file and encode as base64
        let data = std::fs::read(&path)
            .map_err(|e| PyException::new_err(format!("Failed to read file: {}", e)))?;
        let data = base64::engine::general_purpose::STANDARD.encode(&data);

        match llm {
            Some(models) => match models {
                SupportedModels::Claude35HaikuLatest => Ok(Self {
                    content_type: content_type.to_string(),
                    source: Some(DocumentSourceContent {
                        content_type: "base64".to_string(),
                        media_type: media_type.to_string(),
                        data: data.to_string(),
                    }),
                    file_data: None,
                    filename: None,
                }),
                SupportedModels::GPT41Nano => Ok(Self {
                    content_type: {
                        if content_type == "document" {
                            "input_file".to_string()
                        } else if content_type == "image" {
                            "input_image".to_string()
                        } else {
                            content_type.to_string()
                        }
                    },
                    source: None,
                    file_data: Some(format!("data:{};base64,{}", media_type, data.to_string())),
                    filename: Some(file_name),
                }),

                _ => Err(PyException::new_err(
                    "Unsupported LLM for processing document",
                )),
            },
            None => {
                // default using Claude
                Ok(Self {
                    content_type: content_type.to_string(),
                    source: Some(DocumentSourceContent {
                        content_type: "base64".to_string(),
                        media_type: media_type.to_string(),
                        data: data.to_string(),
                    }),
                    file_data: None,
                    filename: None,
                })
            }
        }
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{self:?}"))
    }
}

#[derive(Serialize, Clone, Debug)]
#[pyclass(dict, get_all, set_all, subclass)]
pub struct TextContent {
    #[serde(rename = "type")]
    pub(crate) content_type: String,
    pub(crate) text: String,
}

#[pymethods]
impl TextContent {
    #[new]
    #[pyo3(signature = (text))]
    pub fn new(text: &str) -> PyResult<Self> {
        Ok(TextContent {
            content_type: "text".to_string(),
            text: text.to_string(),
        })
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{self:?}"))
    }
}

#[derive(Serialize, Clone, Debug, IntoPyObject, FromPyObject)]
pub enum ContentTypeInner {
    Document(DocumentContent),
    Text(TextContent),
}

impl ContentTypeInner {
    pub(crate) fn __repr__(&self) -> String {
        match self {
            ContentTypeInner::Document(doc) => format!("{:?}", doc.__repr__().unwrap().to_string()),
            ContentTypeInner::Text(text) => format!("{:?}", text.__repr__().unwrap().to_string()),
        }
    }
}

#[derive(Clone, Debug)]
#[pyclass(dict, get_all, set_all)]
pub struct Content {
    pub ctx: ContentTypeInner,
}

impl Serialize for Content {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match &self.ctx {
            ContentTypeInner::Document(doc) => doc.serialize(serializer),
            ContentTypeInner::Text(text) => text.serialize(serializer),
        }
    }
}

#[pymethods]
impl Content {
    #[new]
    fn new<'p>(content: Bound<'p, PyAny>) -> PyResult<Self> {
        if let Ok(text) = content.extract::<TextContent>() {
            Ok(Self {
                ctx: ContentTypeInner::Text(text),
            })
        } else if let Ok(doc) = content.extract::<DocumentContent>() {
            Ok(Self {
                ctx: ContentTypeInner::Document(doc),
            })
        } else {
            Err(PyTypeError::new_err("Invalid content type"))
        }
    }

    #[classmethod]
    #[pyo3(signature = (text))]
    fn from_text(_cls: Bound<'_, PyType>, text: &str) -> PyResult<Self> {
        Ok(Self {
            ctx: ContentTypeInner::Text(TextContent::new(text)?),
        })
    }

    #[classmethod]
    #[pyo3(signature = (path, llm=None))]
    fn from_document<'p>(_cls: Bound<'p, PyType>, path: &str, llm: Option<&str>) -> PyResult<Self> {
        let _llm = Some(match llm {
            None => SupportedModels::Claude35HaikuLatest,
            _ => SupportedModels::from_str(llm.unwrap()).unwrap(),
        });
        Ok(Self {
            ctx: ContentTypeInner::Document(DocumentContent::new(path, _llm)?),
        })
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.ctx))
    }
}

#[derive(Serialize, Clone, Debug)]
#[pyclass(dict, get_all, set_all, subclass)]
pub struct Message {
    pub(crate) role: String,
    pub(crate) content: Vec<Content>,
}

#[pymethods]
impl Message {
    #[new]
    fn new(content: Vec<Content>) -> Self {
        Self {
            role: "user".to_string(),
            content,
        }
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{self:?}"))
    }
}
