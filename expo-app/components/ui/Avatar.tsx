import React from 'react';
import { View, Image, Text, StyleSheet, ViewStyle } from 'react-native';

type AvatarProps = {
  source?: string;
  fallback?: string;
  size?: 'sm' | 'md' | 'lg';
  style?: ViewStyle;
};

export const Avatar = ({ source, fallback, size = 'md', style }: AvatarProps) => {
  const sizeStyle = {
    sm: styles.small,
    md: styles.medium,
    lg: styles.large,
  }[size];

  const textSizeStyle = {
    sm: styles.smallText,
    md: styles.mediumText,
    lg: styles.largeText,
  }[size];

  return (
    <View style={[styles.container, sizeStyle, style]}>
      {source ? (
        <Image source={{ uri: source }} style={styles.image} />
      ) : (
        <View style={styles.fallbackContainer}>
          <Text style={[styles.fallback, textSizeStyle]}>{fallback || '?'}</Text>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    borderRadius: 100,
    overflow: 'hidden',
    backgroundColor: '#334155', // slate-700
    alignItems: 'center',
    justifyContent: 'center',
  },
  image: {
    width: '100%',
    height: '100%',
  },
  fallbackContainer: {
    width: '100%',
    height: '100%',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#334155', // slate-700
  },
  fallback: {
    color: '#e2e8f0', // slate-200
    fontWeight: '600',
  },
  small: {
    width: 24,
    height: 24,
  },
  medium: {
    width: 40,
    height: 40,
  },
  large: {
    width: 64,
    height: 64,
  },
  smallText: {
    fontSize: 10,
  },
  mediumText: {
    fontSize: 16,
  },
  largeText: {
    fontSize: 24,
  },
}); 