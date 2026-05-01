/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  experimental: {
    typedRoutes: true,
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${process.env.LATENTSPEC_API_BASE || "http://localhost:8000"}/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
