import { Animated, Easing } from 'react-native';

/**
 * Advanced animations utility library for PetPath mobile app
 * Provides spring animations, micro-interactions, and gesture-driven animations
 */

// ========================================
// Spring Animations
// ========================================

export const springAnimation = (
  value: Animated.Value,
  toValue: number,
  config?: {
    friction?: number;
    tension?: number;
    speed?: number;
    bounciness?: number;
  }
): Animated.CompositeAnimation => {
  return Animated.spring(value, {
    toValue,
    friction: config?.friction ?? 7,
    tension: config?.tension ?? 40,
    speed: config?.speed ?? 12,
    bounciness: config?.bounciness ?? 8,
    useNativeDriver: true,
  });
};

export const lightSpring = (
  value: Animated.Value,
  toValue: number
): Animated.CompositeAnimation => {
  return springAnimation(value, toValue, {
    friction: 10,
    tension: 50,
    speed: 15,
    bounciness: 4,
  });
};

export const bouncySpring = (
  value: Animated.Value,
  toValue: number
): Animated.CompositeAnimation => {
  return springAnimation(value, toValue, {
    friction: 5,
    tension: 40,
    speed: 12,
    bounciness: 15,
  });
};

// ========================================
// Timing Animations
// ========================================

export const timingAnimation = (
  value: Animated.Value,
  toValue: number,
  duration: number = 300,
  easing: ((value: number) => number) = Easing.inOut(Easing.ease)
): Animated.CompositeAnimation => {
  return Animated.timing(value, {
    toValue,
    duration,
    easing,
    useNativeDriver: true,
  });
};

export const fadeIn = (
  value: Animated.Value,
  duration: number = 300
): Animated.CompositeAnimation => {
  return timingAnimation(value, 1, duration, Easing.out(Easing.ease));
};

export const fadeOut = (
  value: Animated.Value,
  duration: number = 300
): Animated.CompositeAnimation => {
  return timingAnimation(value, 0, duration, Easing.in(Easing.ease));
};

export const slideIn = (
  value: Animated.Value,
  duration: number = 300
): Animated.CompositeAnimation => {
  return timingAnimation(value, 0, duration, Easing.out(Easing.cubic));
};

export const slideOut = (
  value: Animated.Value,
  toValue: number,
  duration: number = 300
): Animated.CompositeAnimation => {
  return timingAnimation(value, toValue, duration, Easing.in(Easing.cubic));
};

// ========================================
// Scale Animations (Micro-interactions)
// ========================================

export const scalePress = (scale: Animated.Value): void => {
  springAnimation(scale, 0.95, {
    friction: 8,
    tension: 100,
  }).start();
};

export const scaleRelease = (scale: Animated.Value): void => {
  springAnimation(scale, 1, {
    friction: 6,
    tension: 80,
    bounciness: 10,
  }).start();
};

export const pulseAnimation = (
  scale: Animated.Value,
  duration: number = 1000
): Animated.CompositeAnimation => {
  return Animated.loop(
    Animated.sequence([
      Animated.timing(scale, {
        toValue: 1.05,
        duration: duration / 2,
        easing: Easing.inOut(Easing.ease),
        useNativeDriver: true,
      }),
      Animated.timing(scale, {
        toValue: 1,
        duration: duration / 2,
        easing: Easing.inOut(Easing.ease),
        useNativeDriver: true,
      }),
    ])
  );
};

// ========================================
// Rotation Animations
// ========================================

export const rotateAnimation = (
  rotation: Animated.Value,
  toValue: number,
  duration: number = 300
): Animated.CompositeAnimation => {
  return Animated.timing(rotation, {
    toValue,
    duration,
    easing: Easing.linear,
    useNativeDriver: true,
  });
};

export const spinAnimation = (
  rotation: Animated.Value,
  duration: number = 1000
): Animated.CompositeAnimation => {
  return Animated.loop(
    Animated.timing(rotation, {
      toValue: 1,
      duration,
      easing: Easing.linear,
      useNativeDriver: true,
    })
  );
};

// ========================================
// Combined Animations
// ========================================

export const fadeInSlide = (
  opacity: Animated.Value,
  translateY: Animated.Value,
  duration: number = 300
): Animated.CompositeAnimation => {
  return Animated.parallel([
    fadeIn(opacity, duration),
    slideIn(translateY, duration),
  ]);
};

export const fadeOutSlide = (
  opacity: Animated.Value,
  translateY: Animated.Value,
  toValue: number = 50,
  duration: number = 300
): Animated.CompositeAnimation => {
  return Animated.parallel([
    fadeOut(opacity, duration),
    slideOut(translateY, toValue, duration),
  ]);
};

export const scaleInFade = (
  scale: Animated.Value,
  opacity: Animated.Value,
  duration: number = 300
): Animated.CompositeAnimation => {
  return Animated.parallel([
    springAnimation(scale, 1, {
      friction: 8,
      tension: 100,
    }),
    fadeIn(opacity, duration),
  ]);
};

// ========================================
// Stagger Animations
// ========================================

export const staggerAnimation = (
  animations: Animated.CompositeAnimation[],
  staggerDelay: number = 50
): Animated.CompositeAnimation => {
  return Animated.stagger(staggerDelay, animations);
};

export const staggerFadeIn = (
  values: Animated.Value[],
  staggerDelay: number = 50,
  duration: number = 300
): Animated.CompositeAnimation => {
  return staggerAnimation(
    values.map((value) => fadeIn(value, duration)),
    staggerDelay
  );
};

// ========================================
// Progress Animations
// ========================================

export const progressAnimation = (
  progress: Animated.Value,
  toValue: number,
  duration: number = 500
): Animated.CompositeAnimation => {
  return Animated.timing(progress, {
    toValue,
    duration,
    easing: Easing.out(Easing.cubic),
    useNativeDriver: false, // Can't use native driver for width/height
  });
};

export const smoothProgressAnimation = (
  progress: Animated.Value,
  toValue: number,
  duration: number = 800
): Animated.CompositeAnimation => {
  return Animated.timing(progress, {
    toValue,
    duration,
    easing: Easing.bezier(0.25, 0.1, 0.25, 1),
    useNativeDriver: false,
  });
};

// ========================================
// Gesture-driven Animations
// ========================================

export const swipeAnimation = (
  translateX: Animated.Value,
  toValue: number,
  velocity: number = 0
): Animated.CompositeAnimation => {
  const absVelocity = Math.abs(velocity);
  const duration = absVelocity > 0 ? 300 / (absVelocity / 1000) : 300;

  return Animated.timing(translateX, {
    toValue,
    duration: Math.min(duration, 500),
    easing: Easing.out(Easing.cubic),
    useNativeDriver: true,
  });
};

export const rubberBandAnimation = (
  value: Animated.Value,
  maxValue: number
): Animated.CompositeAnimation => {
  return Animated.sequence([
    springAnimation(value, maxValue, {
      friction: 5,
      tension: 80,
      bounciness: 20,
    }),
    springAnimation(value, 0, {
      friction: 7,
      tension: 40,
    }),
  ]);
};

// ========================================
// Loading Animations
// ========================================

export const shimmerAnimation = (
  translateX: Animated.Value,
  width: number,
  duration: number = 1500
): Animated.CompositeAnimation => {
  return Animated.loop(
    Animated.timing(translateX, {
      toValue: width,
      duration,
      easing: Easing.linear,
      useNativeDriver: true,
    })
  );
};

export const dotsAnimation = (
  values: [Animated.Value, Animated.Value, Animated.Value],
  duration: number = 600
): Animated.CompositeAnimation => {
  const [dot1, dot2, dot3] = values;

  const animateDot = (value: Animated.Value, delay: number) =>
    Animated.sequence([
      Animated.delay(delay),
      Animated.timing(value, {
        toValue: -10,
        duration: duration / 2,
        easing: Easing.inOut(Easing.ease),
        useNativeDriver: true,
      }),
      Animated.timing(value, {
        toValue: 0,
        duration: duration / 2,
        easing: Easing.inOut(Easing.ease),
        useNativeDriver: true,
      }),
    ]);

  return Animated.loop(
    Animated.parallel([
      animateDot(dot1, 0),
      animateDot(dot2, duration / 3),
      animateDot(dot3, (duration * 2) / 3),
    ])
  );
};

// ========================================
// Value Interpolations
// ========================================

export const createOpacityInterpolation = (
  value: Animated.Value,
  inputRange: number[] = [0, 1],
  outputRange: number[] = [0, 1]
): Animated.AnimatedInterpolation => {
  return value.interpolate({
    inputRange,
    outputRange,
    extrapolate: 'clamp',
  });
};

export const createScaleInterpolation = (
  value: Animated.Value,
  inputRange: number[] = [0, 1],
  outputRange: number[] = [0, 1]
): Animated.AnimatedInterpolation => {
  return value.interpolate({
    inputRange,
    outputRange,
    extrapolate: 'clamp',
  });
};

export const createRotationInterpolation = (
  value: Animated.Value,
  rotations: number = 1
): Animated.AnimatedInterpolation => {
  return value.interpolate({
    inputRange: [0, 1],
    outputRange: ['0deg', `${360 * rotations}deg`],
  });
};

export const createTranslateInterpolation = (
  value: Animated.Value,
  distance: number,
  inputRange: number[] = [0, 1]
): Animated.AnimatedInterpolation => {
  return value.interpolate({
    inputRange,
    outputRange: [0, distance],
    extrapolate: 'clamp',
  });
};

// ========================================
// Animation Hooks
// ========================================

export const createAnimatedValue = (initialValue: number = 0): Animated.Value => {
  return new Animated.Value(initialValue);
};

export const createAnimatedValueXY = (
  x: number = 0,
  y: number = 0
): Animated.ValueXY => {
  return new Animated.ValueXY({ x, y });
};

// ========================================
// Easing Presets
// ========================================

export const easingPresets = {
  // Standard easings
  linear: Easing.linear,
  ease: Easing.ease,

  // Quadratic
  easeIn: Easing.in(Easing.quad),
  easeOut: Easing.out(Easing.quad),
  easeInOut: Easing.inOut(Easing.quad),

  // Cubic
  cubicIn: Easing.in(Easing.cubic),
  cubicOut: Easing.out(Easing.cubic),
  cubicInOut: Easing.inOut(Easing.cubic),

  // Elastic
  elasticIn: Easing.in(Easing.elastic(1)),
  elasticOut: Easing.out(Easing.elastic(1)),

  // Back
  backIn: Easing.in(Easing.back(1.5)),
  backOut: Easing.out(Easing.back(1.5)),

  // Bezier (iOS-like)
  iosCurve: Easing.bezier(0.25, 0.1, 0.25, 1),
  materialCurve: Easing.bezier(0.4, 0.0, 0.2, 1),
};

// ========================================
// Animation Configs
// ========================================

export const animationConfigs = {
  // Quick micro-interactions
  quickPress: {
    duration: 100,
    easing: easingPresets.easeOut,
  },

  // Standard UI transitions
  standard: {
    duration: 300,
    easing: easingPresets.iosCurve,
  },

  // Smooth page transitions
  pageTransition: {
    duration: 400,
    easing: easingPresets.materialCurve,
  },

  // Spring configs
  springGentle: {
    friction: 10,
    tension: 50,
  },
  springBouncy: {
    friction: 5,
    tension: 40,
    bounciness: 15,
  },
  springSnappy: {
    friction: 8,
    tension: 100,
  },
};

// ========================================
// Utility Functions
// ========================================

export const resetAnimatedValue = (value: Animated.Value, toValue: number = 0): void => {
  value.setValue(toValue);
};

export const resetAnimatedValues = (
  values: Animated.Value[],
  toValue: number = 0
): void => {
  values.forEach((value) => value.setValue(toValue));
};

export const stopAnimation = (
  animation: Animated.CompositeAnimation
): void => {
  animation.stop();
};

export const startAnimation = (
  animation: Animated.CompositeAnimation,
  callback?: Animated.EndCallback
): void => {
  animation.start(callback);
};
