import React from 'react';
import { View, StyleSheet, Text, ScrollView } from 'react-native';
import { SwingMetrics as SwingMetricsType } from '../services/analysisService';
import { LinearGradient } from 'expo-linear-gradient';

interface SwingMetricsProps {
  metrics: SwingMetricsType;
  strengths?: string[];
  weaknesses?: string[];
  suggestions?: string[];
  swingType?: string;
  score?: number;
}

const MetricBar = ({ 
  label, 
  value, 
  maxValue = 100, 
  unit = '', 
  colorStart = '#4CAF50', 
  colorEnd = '#388E3C' 
}: { 
  label: string; 
  value: number; 
  maxValue?: number; 
  unit?: string; 
  colorStart?: string;
  colorEnd?: string;
}) => {
  const percentage = Math.min(100, (value / maxValue) * 100);
  
  return (
    <View style={styles.metricContainer}>
      <View style={styles.metricLabelContainer}>
        <Text style={styles.metricLabel}>{label}</Text>
        <Text style={styles.metricValue}>
          {value.toFixed(1)}{unit}
        </Text>
      </View>
      <View style={styles.barContainer}>
        <LinearGradient
          colors={[colorStart, colorEnd]}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 0 }}
          style={[styles.bar, { width: `${percentage}%` }]}
        />
      </View>
    </View>
  );
};

const SwingMetrics: React.FC<SwingMetricsProps> = ({
  metrics,
  strengths = [],
  weaknesses = [],
  suggestions = [],
  swingType = 'Forehand',
  score
}) => {
  const getScoreColor = (score: number) => {
    if (score >= 80) return ['#4CAF50', '#388E3C'];
    if (score >= 60) return ['#FFC107', '#FFA000'];
    return ['#F44336', '#D32F2F'];
  };

  return (
    <ScrollView style={styles.container}>
      {/* Header with score */}
      <View style={styles.header}>
        <View style={styles.swingTypeContainer}>
          <Text style={styles.swingType}>{swingType}</Text>
        </View>
        
        {score !== undefined && (
          <View style={styles.scoreContainer}>
            <Text style={styles.scoreLabel}>SCORE</Text>
            <LinearGradient
              colors={getScoreColor(score)}
              style={styles.scoreCircle}
            >
              <Text style={styles.scoreText}>{Math.round(score)}</Text>
            </LinearGradient>
          </View>
        )}
      </View>

      {/* Metrics */}
      <View style={styles.metricsContainer}>
        <Text style={styles.sectionTitle}>Metrics</Text>
        <MetricBar 
          label="Racket Speed" 
          value={metrics.racketSpeed} 
          maxValue={120} 
          unit=" mph"
          colorStart="#2196F3"
          colorEnd="#1976D2"
        />
        <MetricBar 
          label="Hip Rotation" 
          value={metrics.hipRotation} 
          maxValue={90} 
          unit="°"
          colorStart="#9C27B0"
          colorEnd="#7B1FA2"
        />
        <MetricBar 
          label="Shoulder Rotation" 
          value={metrics.shoulderRotation} 
          maxValue={120} 
          unit="°"
          colorStart="#FF9800"
          colorEnd="#F57C00"
        />
        <MetricBar 
          label="Knee Flexion" 
          value={metrics.kneeFlexion} 
          maxValue={60} 
          unit="°"
          colorStart="#4CAF50"
          colorEnd="#388E3C"
        />
        <MetricBar 
          label="Weight Transfer" 
          value={metrics.weightTransfer} 
          unit="%"
          colorStart="#FF5722"
          colorEnd="#E64A19"
        />
        <MetricBar 
          label="Balance Score" 
          value={metrics.balanceScore}
          colorStart="#00BCD4"
          colorEnd="#0097A7"
        />
        <MetricBar 
          label="Follow Through" 
          value={metrics.followThrough}
          colorStart="#673AB7"
          colorEnd="#512DA8"
        />
        <MetricBar 
          label="Consistency" 
          value={metrics.consistency}
          colorStart="#FFC107"
          colorEnd="#FFA000"
        />
      </View>

      {/* Strengths */}
      {strengths.length > 0 && (
        <View style={styles.feedbackContainer}>
          <Text style={styles.sectionTitle}>Strengths</Text>
          {strengths.map((strength, index) => (
            <View key={`strength-${index}`} style={styles.feedbackItem}>
              <Text style={styles.bulletPoint}>•</Text>
              <Text style={styles.feedbackText}>{strength}</Text>
            </View>
          ))}
        </View>
      )}

      {/* Weaknesses */}
      {weaknesses.length > 0 && (
        <View style={styles.feedbackContainer}>
          <Text style={styles.sectionTitle}>Areas for Improvement</Text>
          {weaknesses.map((weakness, index) => (
            <View key={`weakness-${index}`} style={styles.feedbackItem}>
              <Text style={styles.bulletPoint}>•</Text>
              <Text style={styles.feedbackText}>{weakness}</Text>
            </View>
          ))}
        </View>
      )}

      {/* Suggestions */}
      {suggestions.length > 0 && (
        <View style={styles.feedbackContainer}>
          <Text style={styles.sectionTitle}>Coach Suggestions</Text>
          {suggestions.map((suggestion, index) => (
            <View key={`suggestion-${index}`} style={styles.feedbackItem}>
              <Text style={styles.bulletPoint}>→</Text>
              <Text style={styles.feedbackText}>{suggestion}</Text>
            </View>
          ))}
        </View>
      )}
      
      {/* Bottom padding */}
      <View style={{ height: 40 }} />
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5',
    padding: 16,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
  },
  swingTypeContainer: {
    backgroundColor: '#333',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
  },
  swingType: {
    color: '#FFF',
    fontWeight: 'bold',
    fontSize: 16,
  },
  scoreContainer: {
    alignItems: 'center',
  },
  scoreLabel: {
    fontSize: 12,
    fontWeight: 'bold',
    color: '#666',
    marginBottom: 4,
  },
  scoreCircle: {
    width: 50,
    height: 50,
    borderRadius: 25,
    justifyContent: 'center',
    alignItems: 'center',
  },
  scoreText: {
    color: '#FFF',
    fontWeight: 'bold',
    fontSize: 18,
  },
  metricsContainer: {
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 12,
    marginTop: 8,
  },
  metricContainer: {
    marginBottom: 12,
  },
  metricLabelContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 4,
  },
  metricLabel: {
    fontSize: 14,
    color: '#333',
  },
  metricValue: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#333',
  },
  barContainer: {
    height: 10,
    backgroundColor: '#E0E0E0',
    borderRadius: 5,
    overflow: 'hidden',
  },
  bar: {
    height: '100%',
    borderRadius: 5,
  },
  feedbackContainer: {
    marginBottom: 20,
  },
  feedbackItem: {
    flexDirection: 'row',
    marginBottom: 8,
    paddingHorizontal: 8,
  },
  bulletPoint: {
    fontSize: 16,
    color: '#333',
    marginRight: 8,
    width: 10,
  },
  feedbackText: {
    fontSize: 14,
    color: '#333',
    flex: 1,
    lineHeight: 20,
  },
});

export default SwingMetrics;