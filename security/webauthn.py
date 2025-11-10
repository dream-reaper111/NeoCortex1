"""Minimal WebAuthn helper wrappers for the dashboard."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Dict, Iterable

from webauthn import verify_authentication_response, verify_registration_response
from webauthn.helpers import (
    generate_authentication_options,
    generate_registration_options,
    options_to_json,
)
from webauthn.helpers.structs import (
    AuthenticationCredential,
    AuthenticationCredentialJSON,
    AuthenticatorSelectionCriteria,
    AuthenticatorTransport,
    PublicKeyCredentialDescriptor,
    RegistrationCredential,
    RegistrationCredentialJSON,
    ResidentKeyRequirement,
    UserVerificationRequirement,
)


@dataclass
class WebAuthnCredential:
    credential_id: str  # base64url encoded
    public_key: str
    sign_count: int

    def descriptor(self) -> PublicKeyCredentialDescriptor:
        padding = "=" * (-len(self.credential_id) % 4)
        return PublicKeyCredentialDescriptor(
            id=base64.urlsafe_b64decode(self.credential_id + padding),
            transports=[AuthenticatorTransport.USB, AuthenticatorTransport.INTERNAL],
        )


def _decode_descriptor(credential_id: str) -> PublicKeyCredentialDescriptor:
    padding = "=" * (-len(credential_id) % 4)
    return PublicKeyCredentialDescriptor(
        id=base64.urlsafe_b64decode(credential_id + padding),
        transports=[AuthenticatorTransport.USB, AuthenticatorTransport.INTERNAL],
    )


def registration_options(
    rp_id: str,
    rp_name: str,
    user_id: str,
    username: str,
    existing_credentials: Iterable[WebAuthnCredential] | None = None,
) -> Dict[str, object]:
    descriptors = [cred.descriptor() for cred in existing_credentials or []]
    options = generate_registration_options(
        rp_id=rp_id,
        rp_name=rp_name,
        user_id=user_id.encode("utf-8"),
        user_name=username,
        user_display_name=username,
        authenticator_selection=AuthenticatorSelectionCriteria(
            resident_key=ResidentKeyRequirement.PREFERRED,
            user_verification=UserVerificationRequirement.REQUIRED,
        ),
        exclude_credentials=descriptors,
    )
    return options_to_json(options)


def authentication_options(
    rp_id: str,
    allow_credentials: Iterable[str] | None = None,
) -> Dict[str, object]:
    descriptors = [_decode_descriptor(cred_id) for cred_id in allow_credentials or []]
    options = generate_authentication_options(
        rp_id=rp_id,
        allow_credentials=descriptors,
        user_verification=UserVerificationRequirement.REQUIRED,
    )
    return options_to_json(options)


def verify_registration(
    credential: RegistrationCredentialJSON,
    expected_rp_id: str,
    expected_origin: str,
    challenge: str,
) -> WebAuthnCredential:
    verification = verify_registration_response(
        credential=RegistrationCredential.parse_obj(credential.dict()),
        expected_challenge=challenge,
        expected_rp_id=expected_rp_id,
        expected_origin=expected_origin,
    )
    credential_id = base64.urlsafe_b64encode(verification.credential_id).decode("utf-8").rstrip("=")
    return WebAuthnCredential(
        credential_id=credential_id,
        public_key=verification.credential_public_key,
        sign_count=verification.sign_count,
    )


def verify_authentication(
    credential: AuthenticationCredentialJSON,
    stored: WebAuthnCredential,
    expected_rp_id: str,
    expected_origin: str,
    challenge: str,
) -> None:
    verify_authentication_response(
        credential=AuthenticationCredential.parse_obj(credential.dict()),
        expected_challenge=challenge,
        expected_rp_id=expected_rp_id,
        expected_origin=expected_origin,
        credential_public_key=stored.public_key,
        credential_current_sign_count=stored.sign_count,
        require_user_verification=True,
    )


__all__ = [
    "WebAuthnCredential",
    "registration_options",
    "authentication_options",
    "verify_registration",
    "verify_authentication",
]
