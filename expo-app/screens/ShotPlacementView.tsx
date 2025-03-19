import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Card } from '../components/ui/Card';

export default function ShotPlacementView() {
  return (
    <View style={styles.container}>
      <Card style={styles.card}>
        <Text style={styles.cardTitle}>Shot Distribution</Text>
        <View style={styles.courtContainer}>
          <View style={styles.court}>
            <View style={styles.courtOutline}>
              <View style={styles.courtCenter}>
                <View style={styles.centerLine} />
                <View style={styles.serviceArea} />
                <View style={styles.serviceBox} />
                <View style={styles.serviceBox} />
              </View>
              <View style={styles.baselineDots}>
                {/* Dots showing where shots landed */}
                <View style={[styles.dot, styles.dotForehand, { left: '70%', top: '40%' }]} />
                <View style={[styles.dot, styles.dotBackhand, { left: '25%', top: '30%' }]} />
                <View style={[styles.dot, styles.dotForehand, { left: '85%', top: '65%' }]} />
                <View style={[styles.dot, styles.dotBackhand, { left: '15%', top: '60%' }]} />
                <View style={[styles.dot, styles.dotForehand, { left: '60%', top: '80%' }]} />
                <View style={[styles.dot, styles.dotForehand, { left: '50%', top: '20%' }]} />
                <View style={[styles.dot, styles.dotBackhand, { left: '30%', top: '75%' }]} />
              </View>
            </View>
          </View>
        </View>
      </Card>

      <Card style={styles.card}>
        <Text style={styles.cardTitle}>Shot Analytics</Text>
        
        <View style={styles.statItem}>
          <Text style={styles.statLabel}>Forehand Placement</Text>
          <View style={styles.statRow}>
            <View style={styles.statBlock}>
              <Text style={styles.statValue}>38%</Text>
              <Text style={styles.statDesc}>Cross Court</Text>
            </View>
            <View style={styles.statBlock}>
              <Text style={styles.statValue}>42%</Text>
              <Text style={styles.statDesc}>Down the Line</Text>
            </View>
            <View style={styles.statBlock}>
              <Text style={styles.statValue}>20%</Text>
              <Text style={styles.statDesc}>Middle</Text>
            </View>
          </View>
        </View>

        <View style={styles.statItem}>
          <Text style={styles.statLabel}>Backhand Placement</Text>
          <View style={styles.statRow}>
            <View style={styles.statBlock}>
              <Text style={styles.statValue}>52%</Text>
              <Text style={styles.statDesc}>Cross Court</Text>
            </View>
            <View style={styles.statBlock}>
              <Text style={styles.statValue}>28%</Text>
              <Text style={styles.statDesc}>Down the Line</Text>
            </View>
            <View style={styles.statBlock}>
              <Text style={styles.statValue}>20%</Text>
              <Text style={styles.statDesc}>Middle</Text>
            </View>
          </View>
        </View>
      </Card>

      <Card style={styles.card}>
        <Text style={styles.cardTitle}>Shot Depth Analysis</Text>
        <View style={styles.depthBars}>
          <View style={styles.depthBar}>
            <Text style={styles.depthLabel}>Serve</Text>
            <View style={styles.depthBarOuter}>
              <View style={[styles.depthBarInner, { width: '85%' }]} />
            </View>
            <Text style={styles.depthValue}>85%</Text>
          </View>
          <View style={styles.depthBar}>
            <Text style={styles.depthLabel}>Forehand</Text>
            <View style={styles.depthBarOuter}>
              <View style={[styles.depthBarInner, { width: '78%' }]} />
            </View>
            <Text style={styles.depthValue}>78%</Text>
          </View>
          <View style={styles.depthBar}>
            <Text style={styles.depthLabel}>Backhand</Text>
            <View style={styles.depthBarOuter}>
              <View style={[styles.depthBarInner, { width: '68%' }]} />
            </View>
            <Text style={styles.depthValue}>68%</Text>
          </View>
        </View>
      </Card>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
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
  courtContainer: {
    alignItems: 'center',
    marginVertical: 16,
  },
  court: {
    width: '100%',
    aspectRatio: 2 / 1,
    maxWidth: 300,
    backgroundColor: '#1e293b', // slate-800
    borderRadius: 4,
    padding: 8,
  },
  courtOutline: {
    flex: 1,
    borderWidth: 1,
    borderColor: '#94a3b8', // slate-400
    position: 'relative',
  },
  courtCenter: {
    flex: 1,
    borderBottomWidth: 1,
    borderColor: '#94a3b8', // slate-400
  },
  centerLine: {
    position: 'absolute',
    height: '100%',
    width: 1,
    backgroundColor: '#94a3b8', // slate-400
    left: '50%',
  },
  serviceArea: {
    position: 'absolute',
    top: '35%',
    left: 0,
    right: 0,
    height: 1,
    backgroundColor: '#94a3b8', // slate-400
  },
  serviceBox: {
    height: '35%',
    width: '50%',
    borderRightWidth: 1,
    borderColor: '#94a3b8', // slate-400
  },
  baselineDots: {
    position: 'absolute',
    width: '100%',
    height: '100%',
  },
  dot: {
    position: 'absolute',
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  dotForehand: {
    backgroundColor: '#6366f1', // indigo-500
  },
  dotBackhand: {
    backgroundColor: '#3b82f6', // blue-500
  },
  statItem: {
    marginBottom: 16,
  },
  statLabel: {
    color: '#e2e8f0', // slate-200
    marginBottom: 8,
  },
  statRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  statBlock: {
    flex: 1,
    alignItems: 'center',
    backgroundColor: '#1e293b', // slate-800
    paddingVertical: 8,
    borderRadius: 4,
    marginHorizontal: 2,
  },
  statValue: {
    color: '#a5b4fc', // indigo-300
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  statDesc: {
    color: '#94a3b8', // slate-400
    fontSize: 12,
  },
  depthBars: {
    marginTop: 8,
  },
  depthBar: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  depthLabel: {
    width: 80,
    color: '#e2e8f0', // slate-200
  },
  depthBarOuter: {
    flex: 1,
    height: 12,
    backgroundColor: '#1e293b', // slate-800
    borderRadius: 6,
    overflow: 'hidden',
    marginHorizontal: 8,
  },
  depthBarInner: {
    height: '100%',
    backgroundColor: '#6366f1', // indigo-500
    borderRadius: 6,
  },
  depthValue: {
    width: 40,
    textAlign: 'right',
    color: '#a5b4fc', // indigo-300
    fontWeight: '600',
  },
}); 