-- PostgreSQL truncates identifiers to 63 bytes. The original Media Library
-- migration supplied a 64-character index name, while Prisma's generated
-- datamodel name truncates at a different boundary. Rename the physical index
-- forward-only so migration history and the Prisma datamodel converge.

ALTER INDEX "media_external_references_owner_id_provider_provider_item_id_id"
RENAME TO "media_external_references_owner_id_provider_provider_item_i_idx";
