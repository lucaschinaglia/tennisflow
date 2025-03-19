import React from 'react';
import { TouchableOpacity, Text, StyleSheet, ViewStyle, TextStyle, StyleProp } from 'react-native';

type ButtonProps = {
  children: React.ReactNode;
  variant?: 'default' | 'ghost' | 'outline';
  size?: 'default' | 'sm' | 'lg' | 'icon';
  className?: string;
  style?: StyleProp<ViewStyle>;
  textStyle?: TextStyle;
  onPress?: () => void;
};

export const Button = ({
  children,
  variant = 'default',
  size = 'default',
  style,
  textStyle,
  onPress,
}: ButtonProps) => {
  const getVariantStyle = () => {
    switch (variant) {
      case 'ghost':
        return styles.ghost;
      case 'outline':
        return styles.outline;
      default:
        return styles.default;
    }
  };

  const getSizeStyle = () => {
    switch (size) {
      case 'sm':
        return styles.small;
      case 'lg':
        return styles.large;
      case 'icon':
        return styles.icon;
      default:
        return styles.defaultSize;
    }
  };

  const getTextStyle = () => {
    switch (variant) {
      case 'ghost':
        return styles.ghostText;
      case 'outline':
        return styles.outlineText;
      default:
        return styles.defaultText;
    }
  };

  return (
    <TouchableOpacity
      style={[styles.button, getVariantStyle(), getSizeStyle(), style]}
      onPress={onPress}
      activeOpacity={0.8}
    >
      {typeof children === 'string' ? (
        <Text style={[styles.text, getTextStyle(), textStyle]}>{children}</Text>
      ) : (
        children
      )}
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  button: {
    borderRadius: 8,
    justifyContent: 'center',
    alignItems: 'center',
    flexDirection: 'row',
  },
  text: {
    fontWeight: '600',
  },
  // Variants
  default: {
    backgroundColor: '#6366f1', // indigo-500
  },
  ghost: {
    backgroundColor: 'transparent',
  },
  outline: {
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderColor: '#6366f1', // indigo-500
  },
  // Sizes
  defaultSize: {
    paddingVertical: 10,
    paddingHorizontal: 16,
  },
  small: {
    paddingVertical: 6,
    paddingHorizontal: 12,
  },
  large: {
    paddingVertical: 12,
    paddingHorizontal: 20,
  },
  icon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    padding: 0,
  },
  // Text colors
  defaultText: {
    color: '#ffffff',
  },
  ghostText: {
    color: '#6366f1', // indigo-500
  },
  outlineText: {
    color: '#6366f1', // indigo-500
  },
}); 