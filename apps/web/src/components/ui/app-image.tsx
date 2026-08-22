import Image, { type ImageProps } from 'next/image';

export type AppImageProps = Omit<ImageProps, 'width' | 'height'> & {
  width?: ImageProps['width'];
  height?: ImageProps['height'];
};

/**
 * Image boundary for user-generated, signed, and externally hosted media.
 *
 * Woof cannot assume every private/signed media origin is known at build time,
 * so optimization is disabled by default while retaining Next's image element,
 * accessibility contract, and explicit intrinsic dimensions.
 */
export function AppImage({
  width = 800,
  height = 600,
  unoptimized = true,
  alt,
  ...props
}: AppImageProps) {
  return (
    <Image
      {...props}
      alt={alt}
      width={width}
      height={height}
      unoptimized={unoptimized}
    />
  );
}
