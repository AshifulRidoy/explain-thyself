import { drizzle } from "drizzle-orm/node-postgres";
import { Pool } from "pg";
import * as schema from "./schema.js";

const pool = new Pool({
  connectionString:
    process.env.DATABASE_URL ??
    "postgresql://ets:ets@localhost:5432/explain_the_self",
  max: 5,
});

export const db = drizzle(pool, { schema });
export { schema };
