import React from 'react';
import { View, StyleSheet, ViewStyle } from 'react-native';

type ProgressProps = {
  value?: number;
  max?: number;
  style?: ViewStyle;
  indicatorStyle?: ViewStyle;
};

export const Progress = ({ 
  value = 0, 
  max = 100, 
  style, 
  indicatorStyle 
}: ProgressProps) => {
  const percentage = Math.min(Math.max(0, value / max), 1) * 100;

  return (
    <View style={[styles.container, style]}>
      <View 
        style={[
          styles.indicator, 
          { width: `${percentage}%` },
          indicatorStyle
        ]} 
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    width: '100%',
    height: 8,
    backgroundColor: '#334155', // slate-700
    borderRadius: 4,
    overflow: 'hidden',
  },
  indicator: {
    height: '100%',
    backgroundColor: '#6366f1', // indigo-500
    borderRadius: 4,
  },
}); 