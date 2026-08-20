CREATE TABLE "counterfactuals" (
	"id" text PRIMARY KEY NOT NULL,
	"trace_id" text NOT NULL,
	"variable" text NOT NULL,
	"original_word" text,
	"replacement_word" text,
	"prompt_text" text NOT NULL,
	"impact" real NOT NULL,
	"payload" jsonb NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
ALTER TABLE "counterfactuals" ADD CONSTRAINT "counterfactuals_trace_id_traces_id_fk" FOREIGN KEY ("trace_id") REFERENCES "public"."traces"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
CREATE INDEX "counterfactuals_trace_idx" ON "counterfactuals" USING btree ("trace_id");