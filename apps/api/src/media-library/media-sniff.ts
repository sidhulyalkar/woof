import { open } from 'node:fs/promises';

export type SniffedMediaType =
  | 'image/jpeg'
  | 'image/png'
  | 'image/webp'
  | 'image/heic'
  | 'video/mp4'
  | 'video/quicktime'
  | 'video/webm';

const HEIF_BRANDS = new Set(['heic', 'heix', 'hevc', 'hevx', 'heim', 'heis', 'mif1', 'msf1']);
const QUICKTIME_BRANDS = new Set(['qt  ']);

export function sniffMediaType(header: Uint8Array): SniffedMediaType | null {
  const bytes = Buffer.from(header);
  if (bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) {
    return 'image/jpeg';
  }
  if (
    bytes.length >= 8 &&
    bytes.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]))
  ) {
    return 'image/png';
  }
  if (
    bytes.length >= 12 &&
    bytes.subarray(0, 4).toString('ascii') === 'RIFF' &&
    bytes.subarray(8, 12).toString('ascii') === 'WEBP'
  ) {
    return 'image/webp';
  }
  if (
    bytes.length >= 4 &&
    bytes[0] === 0x1a &&
    bytes[1] === 0x45 &&
    bytes[2] === 0xdf &&
    bytes[3] === 0xa3
  ) {
    return 'video/webm';
  }
  if (bytes.length >= 12 && bytes.subarray(4, 8).toString('ascii') === 'ftyp') {
    const brand = bytes.subarray(8, 12).toString('ascii');
    if (HEIF_BRANDS.has(brand)) return 'image/heic';
    if (QUICKTIME_BRANDS.has(brand)) return 'video/quicktime';
    return 'video/mp4';
  }
  return null;
}

export function declaredMimeMatches(actual: SniffedMediaType | null, declared: string) {
  if (!actual) return false;
  if (actual === declared) return true;
  return actual === 'image/heic' && (declared === 'image/heic' || declared === 'image/heif');
}

export async function sniffMediaFile(filePath: string, headerBytes = 64) {
  const handle = await open(filePath, 'r');
  try {
    const buffer = Buffer.alloc(headerBytes);
    const { bytesRead } = await handle.read(buffer, 0, headerBytes, 0);
    return sniffMediaType(buffer.subarray(0, bytesRead));
  } finally {
    await handle.close();
  }
}
