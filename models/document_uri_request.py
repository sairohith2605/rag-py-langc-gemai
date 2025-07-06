from pydantic import BaseModel


class DocumentURIRequest(BaseModel):

    uri: str
