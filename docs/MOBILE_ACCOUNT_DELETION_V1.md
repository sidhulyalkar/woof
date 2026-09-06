# Mobile Account Deletion v1

## Purpose

Woof account creation on native clients must be paired with an easy-to-find in-app deletion path. This tranche connects the qualified server-owned `DELETE /users/me` authority to the Profile screen without creating a second deletion protocol in the client.

## User experience

Profile contains a visible **Account → Session and privacy** section with separate Logout and Delete account actions.

Delete account uses two explicit destructive confirmations. The copy explains that deletion permanently removes the Woof account, owned pet profiles and relationship history, and Woof-owned private Media Library files.

The final confirmation also explains the retry boundary: sessions end only after the server confirms deletion. If the server cannot complete deletion, the client keeps the account active for a later retry.

## Client authority

The client performs exactly one destructive server operation:

`DELETE /users/me`

The authenticated server subject remains the account authority. Mobile does not send or choose a user id to delete.

The access token is removed from SecureStore only after the server request resolves successfully. AuthContext clears the local user only after the same success. A retryable failure therefore does not intentionally convert an undeleted server account into a locally logged-out state.

## Qualification

`Mobile Account Deletion CI` proves from a frozen install that:

1. mobile calls the canonical self-delete endpoint;
2. server deletion precedes local token deletion;
3. AuthContext retains user state until server success;
4. Profile retains two destructive confirmations and retry copy;
5. the delete control has an accessibility label;
6. the tranche is canonically formatted;
7. the complete native client type-checks;
8. the complete native client lints with zero warnings.

## Privacy boundary

This UI inherits the server deletion authority's bounded evidence. It must not promise more than the backend proves.

In particular, v1 does not claim immediate physical erasure of historical verification-document bytes whose legacy storage ownership is not yet qualified, external-provider copies, or immutable backups before their retention policy expires.

Those boundaries should remain consistent with `ACCOUNT_DELETION_AUTHORITY_V1.md` and the product privacy policy.

## Explicit non-claims

This tranche does not claim:

- that account deletion has already shipped to TestFlight or the App Store;
- physical-device validation;
- deletion of data outside the qualified server contract;
- Apple review acceptance;
- a stronger historical verification-document erasure guarantee than the server can prove.

## Exit condition

This tranche is complete when Woof can truthfully say:

> A signed-in mobile user can find account deletion in Profile, explicitly confirm it, and have the client end local authentication only after the server confirms the self-owned deletion completed.
