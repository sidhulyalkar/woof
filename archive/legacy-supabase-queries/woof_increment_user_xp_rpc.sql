CREATE OR REPLACE FUNCTION public.increment_user_xp(target_user UUID, delta INT)
RETURNS VOID AS $$
BEGIN
  UPDATE public.users 
  SET xp = GREATEST(0, xp + delta),
      level = GREATEST(1, 1 + (xp + delta) / 100)
  WHERE id = target_user;
END; $$ LANGUAGE plpgsql SECURITY DEFINER;