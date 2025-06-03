use pyo3::{pyclass, pymethods, Bound, FromPyObject, IntoPyObject, PyAny, PyResult};
use pyo3::exceptions::{PyException, PyTypeError};
use pyo3::prelude::PyAnyMethods;
use pyo3::types::{PyDict, PyTuple, PyType};
use serde::{Serialize, Serializer};

#[derive(Serialize, Clone, Debug)]
#[pyclass(dict, get_all, set_all, subclass)]
pub struct DocumentSourceContent {
    #[serde(rename = "type")]
    pub(crate) content_type: String,
    pub(crate) media_type: String,
    pub(crate) data: String, // base64
}

#[derive(Serialize, Clone, Debug)]
#[pyclass(dict, get_all, set_all, subclass)]
pub struct DocumentContent {
    #[serde(rename = "type")]
    pub(crate) content_type: String,
    pub(crate) source: DocumentSourceContent,
}

#[pymethods]
impl DocumentContent {
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
    fn new(text: &str) -> Self {
        Self {
            content_type: "text".to_string(),
            text: text.to_string(),
        }
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
    fn from_text(_cls: Bound<'_, PyType>, text: &str) -> PyResult<Self> {
        Ok(Self {
            ctx: ContentTypeInner::Text(TextContent {
                content_type: "text".to_string(),
                text: text.to_string(),
            }),
        })
    }

    #[allow(unused_variables)]
    #[classmethod]
    #[pyo3(signature = (*args, **kwargs))]
    fn from_document<'p>(
        _cls: Bound<'p, PyType>,
        args: Bound<'p, PyTuple>,
        kwargs: Option<Bound<'p, PyDict>>,
    ) -> PyResult<Self> {
        Err(PyException::new_err("Not implemented yet"))
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