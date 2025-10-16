-- AddForeignKey
ALTER TABLE "location_pings" ADD CONSTRAINT "location_pings_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
