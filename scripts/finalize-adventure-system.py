from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text()
    if old not in text:
        raise RuntimeError(f"Expected patch target not found in {path}")
    file.write_text(text.replace(old, new, 1))


replace_once(
    "apps/api/src/adventure/adventure.service.ts",
    "import { createHash, randomUUID } from 'node:crypto';",
    "import { createHash } from 'node:crypto';",
)

replace_once(
    "apps/web/src/app/journey/page.tsx",
    "{asset.url && asset.mediaType === 'image' ? (\n                        <img",
    "{asset.url && asset.mediaType === 'image' ? (\n                        <>\n                          {/* eslint-disable-next-line @next/next/no-img-element */}\n                          <img",
)
replace_once(
    "apps/web/src/app/journey/page.tsx",
    "className=\"h-full w-full object-cover transition-transform duration-300 group-hover:scale-[1.03]\"\n                        />\n                      ) : (",
    "className=\"h-full w-full object-cover transition-transform duration-300 group-hover:scale-[1.03]\"\n                          />\n                        </>\n                      ) : (",
)

replace_once(
    "apps/web/src/app/health/page.tsx",
    "      <BottomNav />\n    </div>\n  );",
    "      {result?.assessment.triage !== 'emergency_now' && <BottomNav />}\n    </div>\n  );",
)
