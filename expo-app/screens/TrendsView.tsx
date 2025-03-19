import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Card } from '../components/ui/Card';

export default function TrendsView() {
  return (
    <View style={styles.container}>
      <Card style={styles.card}>
        <Text style={styles.cardTitle}>Performance Trends</Text>
        <View style={styles.chartPlaceholder}>
          <Text style={styles.placeholderText}>Performance Chart Visualization</Text>
        </View>
      </Card>

      <Card style={styles.card}>
        <Text style={styles.cardTitle}>Skills Development</Text>
        <View style={styles.chartPlaceholder}>
          <Text style={styles.placeholderText}>Skills Radar Chart</Text>
        </View>
      </Card>

      <Card style={styles.card}>
        <Text style={styles.cardTitle}>Recent Improvements</Text>
        <View style={styles.listItem}>
          <Text style={styles.listItemTitle}>Serve Accuracy</Text>
          <Text style={styles.listItemValue}>+12%</Text>
        </View>
        <View style={styles.listItem}>
          <Text style={styles.listItemTitle}>Forehand Power</Text>
          <Text style={styles.listItemValue}>+8%</Text>
        </View>
        <View style={styles.listItem}>
          <Text style={styles.listItemTitle}>Backhand Consistency</Text>
          <Text style={styles.listItemValue}>+15%</Text>
        </View>
      </Card>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    gap: 16,
  },
  card: {
    marginBottom: 16,
  },
  cardTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
    marginBottom: 12,
  },
  chartPlaceholder: {
    height: 180,
    backgroundColor: '#1e293b', // slate-800
    borderRadius: 8,
    justifyContent: 'center',
    alignItems: 'center',
  },
  placeholderText: {
    color: '#64748b', // slate-500
    fontStyle: 'italic',
  },
  listItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#1e293b', // slate-800
  },
  listItemTitle: {
    color: '#e2e8f0', // slate-200
  },
  listItemValue: {
    color: '#a5b4fc', // indigo-300
    fontWeight: '600',
  },
}); 