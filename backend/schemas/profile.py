from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator


class ProfilePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = Field(default=None, max_length=100)
    date_of_birth: Optional[date] = None
    gender: Optional[str] = Field(default=None, max_length=30)
    occupation_status: Optional[Literal["student", "working", "other"]] = None
    degree: Optional[str] = Field(default=None, max_length=100)
    branch: Optional[str] = Field(default=None, max_length=150)
    study_year: Optional[str] = Field(default=None, max_length=30)
    job_title: Optional[str] = Field(default=None, max_length=150)
    city: Optional[str] = Field(default=None, max_length=100)
    state: Optional[str] = Field(default=None, max_length=100)
    country: Optional[str] = Field(default=None, max_length=100)

    @model_validator(mode="before")
    @classmethod
    def empty_strings_to_none(cls, values):
        if isinstance(values, dict):
            return {key: (None if value == "" else value) for key, value in values.items()}
        return values

    @field_validator("date_of_birth")
    @classmethod
    def validate_birth_date(cls, value):
        if value and value > date.today():
            raise ValueError("Date of birth cannot be in the future")
        return value

    @model_validator(mode="after")
    def validate_occupation_fields(self):
        if self.occupation_status != "student":
            self.degree = None
            self.branch = None
            self.study_year = None
        if self.occupation_status not in ("working", "other"):
            self.job_title = None
        return self


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(min_length=8)
    profile: Optional[ProfilePayload] = None
