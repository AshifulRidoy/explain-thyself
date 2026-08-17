import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  transpilePackages: ["@ets/trace-schema", "@ets/design-tokens"],
};

export default nextConfig;
