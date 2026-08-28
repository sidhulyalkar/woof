-- QuestInteraction is historical evidence. Exact retries may converge on the
-- existing row, but a later request must never rewrite the pathway, context,
-- identity, or original timestamp of an already-recorded interaction.
--
-- The application currently uses `INSERT ... ON CONFLICT DO UPDATE` for retry
-- convergence. This BEFORE UPDATE trigger therefore distinguishes a semantic
-- no-op retry from divergent history:
--   * exact semantic duplicate -> preserve OLD row, including created_at
--   * any changed historical field -> fail closed
--
-- Deletes remain allowed so ordinary account/pet cascade deletion and privacy
-- workflows are not blocked by the immutability contract.

CREATE OR REPLACE FUNCTION enforce_quest_interaction_immutability()
RETURNS TRIGGER AS $$
BEGIN
  IF NEW.id IS NOT DISTINCT FROM OLD.id
     AND NEW.user_id IS NOT DISTINCT FROM OLD.user_id
     AND NEW.pet_id IS NOT DISTINCT FROM OLD.pet_id
     AND NEW.quest_id IS NOT DISTINCT FROM OLD.quest_id
     AND NEW.interaction IS NOT DISTINCT FROM OLD.interaction
     AND NEW.pathway IS NOT DISTINCT FROM OLD.pathway
     AND NEW.context IS NOT DISTINCT FROM OLD.context THEN
    RETURN OLD;
  END IF;

  RAISE EXCEPTION 'quest_interactions are immutable; divergent retry rejected'
    USING ERRCODE = '23514';
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS quest_interactions_immutable ON quest_interactions;

CREATE TRIGGER quest_interactions_immutable
BEFORE UPDATE ON quest_interactions
FOR EACH ROW
EXECUTE FUNCTION enforce_quest_interaction_immutability();
