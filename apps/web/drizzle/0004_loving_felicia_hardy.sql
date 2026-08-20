CREATE TABLE "comparisons" (
	"id" text PRIMARY KEY NOT NULL,
	"trace_id_a" text NOT NULL,
	"trace_id_b" text NOT NULL,
	"model_a" text NOT NULL,
	"model_b" text NOT NULL,
	"agreement" real NOT NULL,
	"payload" jsonb NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
ALTER TABLE "comparisons" ADD CONSTRAINT "comparisons_trace_id_a_traces_id_fk" FOREIGN KEY ("trace_id_a") REFERENCES "public"."traces"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "comparisons" ADD CONSTRAINT "comparisons_trace_id_b_traces_id_fk" FOREIGN KEY ("trace_id_b") REFERENCES "public"."traces"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
CREATE INDEX "comparisons_trace_a_idx" ON "comparisons" USING btree ("trace_id_a");