import { declaredMimeMatches, sniffMediaType } from './media-sniff';

function ftyp(brand: string) {
  const bytes = Buffer.alloc(24);
  bytes.writeUInt32BE(24, 0);
  bytes.write('ftyp', 4, 'ascii');
  bytes.write(brand, 8, 'ascii');
  return bytes;
}

describe('media signature sniffing', () => {
  it.each([
    [Buffer.from([0xff, 0xd8, 0xff, 0xe0]), 'image/jpeg'],
    [Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]), 'image/png'],
    [Buffer.from('RIFF0000WEBP', 'ascii'), 'image/webp'],
    [Buffer.from([0x1a, 0x45, 0xdf, 0xa3]), 'video/webm'],
    [ftyp('isom'), 'video/mp4'],
    [ftyp('qt  '), 'video/quicktime'],
    [ftyp('heic'), 'image/heic'],
  ])('detects %s as %s', (bytes, expected) => {
    expect(sniffMediaType(bytes as Buffer)).toBe(expected);
  });

  it('allows the HEIC/HEIF family but rejects renamed unrelated media', () => {
    expect(declaredMimeMatches('image/heic', 'image/heif')).toBe(true);
    expect(declaredMimeMatches('image/jpeg', 'video/mp4')).toBe(false);
    expect(declaredMimeMatches(null, 'image/jpeg')).toBe(false);
  });
});
