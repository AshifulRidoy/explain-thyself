CREATE TABLE "concepts" (
	"id" text PRIMARY KEY NOT NULL,
	"trace_id" text NOT NULL,
	"label" text NOT NULL,
	"score" real NOT NULL,
	"kind" text DEFAULT 'INTERPRETED' NOT NULL,
	"payload" jsonb
);
--> statement-breakpoint
CREATE TABLE "trace_events" (
	"id" bigserial PRIMARY KEY NOT NULL,
	"trace_id" text NOT NULL,
	"seq" integer NOT NULL,
	"type" text NOT NULL,
	"level" text NOT NULL,
	"position" integer,
	"layer" integer,
	"payload" jsonb NOT NULL,
	"t" integer DEFAULT 0 NOT NULL
);
--> statement-breakpoint
CREATE TABLE "traces" (
	"id" text PRIMARY KEY NOT NULL,
	"display_id" serial NOT NULL,
	"session_id" text,
	"model_name" text NOT NULL,
	"model_revision" text DEFAULT '' NOT NULL,
	"device" text DEFAULT '' NOT NULL,
	"trace_mode" text DEFAULT 'STANDARD' NOT NULL,
	"input" text NOT NULL,
	"output_text" text,
	"status" text DEFAULT 'streaming' NOT NULL,
	"max_tokens" integer DEFAULT 60 NOT NULL,
	"temperature" real DEFAULT 0 NOT NULL,
	"token_count" integer,
	"duration_ms" integer,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	CONSTRAINT "traces_display_id_unique" UNIQUE("display_id")
);
--> statement-breakpoint
ALTER TABLE "concepts" ADD CONSTRAINT "concepts_trace_id_traces_id_fk" FOREIGN KEY ("trace_id") REFERENCES "public"."traces"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "trace_events" ADD CONSTRAINT "trace_events_trace_id_traces_id_fk" FOREIGN KEY ("trace_id") REFERENCES "public"."traces"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
CREATE INDEX "trace_events_trace_seq_idx" ON "trace_events" USING btree ("trace_id","seq");