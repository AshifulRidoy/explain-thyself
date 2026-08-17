import Link from "next/link";

const NAV = [
  { href: "/explore", label: "Explore" },
  { href: "/traces", label: "Traces" },
  { href: "/methodology", label: "Methodology" },
] as const;

export function SiteHeader() {
  return (
    <header className="border-b border-line">
      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-6">
        <Link href="/" className="font-serif text-lg leading-none">
          Explain <span className="italic">The</span> Self
        </Link>
        <nav className="flex items-center gap-6">
          {NAV.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className="text-interface text-muted transition-colors hover:text-ink"
            >
              {item.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
