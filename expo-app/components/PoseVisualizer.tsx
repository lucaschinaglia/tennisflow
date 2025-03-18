import React, { useRef, useEffect } from 'react';
import { View, StyleSheet, Text, Image, Dimensions } from 'react-native';
import Svg, { Circle, Line, Text as SvgText } from 'react-native-svg';
import { PoseData, Annotation } from '../services/analysisService';

interface PoseVisualizerProps {
  imageUri?: string;  // URI of the video frame image (optional)
  poseData: PoseData;
  annotations?: Annotation[];
  swingPhase?: string;
  width?: number;
  height?: number;
}

const DEFAULT_WIDTH = Dimensions.get('window').width;
const DEFAULT_HEIGHT = DEFAULT_WIDTH * 0.75; // 4:3 aspect ratio

const PoseVisualizer: React.FC<PoseVisualizerProps> = ({
  imageUri,
  poseData,
  annotations = [],
  swingPhase,
  width = DEFAULT_WIDTH,
  height = DEFAULT_HEIGHT,
}) => {
  // Ref for saving the view dimensions
  const svgRef = useRef<View>(null);

  // Keypoint size based on canvas dimensions
  const KEYPOINT_RADIUS = width * 0.01;
  const CONNECTION_STROKE_WIDTH = width * 0.005;
  const TEXT_SIZE = width * 0.03;

  // Define colors for different body parts
  const colorMap = {
    left_shoulder: '#FF5722',
    right_shoulder: '#FF5722',
    left_elbow: '#FF9800',
    right_elbow: '#FF9800',
    left_wrist: '#FFC107',
    right_wrist: '#FFC107',
    left_hip: '#4CAF50',
    right_hip: '#4CAF50',
    left_knee: '#2196F3',
    right_knee: '#2196F3',
    left_ankle: '#3F51B5',
    right_ankle: '#3F51B5',
    nose: '#9C27B0',
    left_eye: '#9C27B0',
    right_eye: '#9C27B0',
    left_ear: '#9C27B0',
    right_ear: '#9C27B0',
  };

  // Get color for a keypoint
  const getKeypointColor = (name: string): string => {
    return colorMap[name as keyof typeof colorMap] || '#FF0000';
  };

  // Get coordinates for keypoint in the container's coordinate system
  const getCoordinates = (keypoint: { position: { x: number; y: number } }) => {
    // Convert normalized coordinates (0-1) to pixel coordinates
    return {
      x: keypoint.position.x * width,
      y: keypoint.position.y * height,
    };
  };

  return (
    <View style={[styles.container, { width, height }]}>
      {/* Background image if provided */}
      {imageUri && (
        <Image
          source={{ uri: imageUri }}
          style={[StyleSheet.absoluteFillObject, { width, height }]}
          resizeMode="cover"
        />
      )}

      {/* SVG overlay for pose visualization */}
      <Svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        style={StyleSheet.absoluteFillObject}
        ref={svgRef}
      >
        {/* Draw connections first (so they appear behind keypoints) */}
        {poseData.connections.map((connection, index) => {
          const fromKeypoint = poseData.keypoints.find(k => k.name === connection.from);
          const toKeypoint = poseData.keypoints.find(k => k.name === connection.to);

          if (!fromKeypoint || !toKeypoint) return null;

          const fromCoords = getCoordinates(fromKeypoint);
          const toCoords = getCoordinates(toKeypoint);
          
          // Use average confidence to determine opacity
          const avgConfidence = (fromKeypoint.confidence + toKeypoint.confidence) / 2;
          const opacity = Math.max(0.3, avgConfidence);
          
          return (
            <Line
              key={`connection-${index}`}
              x1={fromCoords.x}
              y1={fromCoords.y}
              x2={toCoords.x}
              y2={toCoords.y}
              stroke="#FFFFFF"
              strokeWidth={CONNECTION_STROKE_WIDTH}
              opacity={opacity}
            />
          );
        })}

        {/* Draw keypoints */}
        {poseData.keypoints.map((keypoint, index) => {
          const coords = getCoordinates(keypoint);
          const color = getKeypointColor(keypoint.name);
          
          return (
            <Circle
              key={`keypoint-${index}`}
              cx={coords.x}
              cy={coords.y}
              r={KEYPOINT_RADIUS}
              fill={color}
              opacity={keypoint.confidence}
              stroke="#FFFFFF"
              strokeWidth={1}
            />
          );
        })}

        {/* Draw annotations */}
        {annotations.map((annotation, index) => {
          const coords = getCoordinates(annotation.position);
          
          return (
            <React.Fragment key={`annotation-${index}`}>
              {/* Circle marker for the annotation point */}
              <Circle
                cx={coords.x}
                cy={coords.y}
                r={KEYPOINT_RADIUS * 1.5}
                fill={annotation.color}
                opacity={0.8}
                stroke="#FFFFFF"
                strokeWidth={1}
              />
              
              {/* Text label */}
              <SvgText
                x={coords.x}
                y={coords.y - KEYPOINT_RADIUS * 3}
                fill="#FFFFFF"
                stroke="#000000"
                strokeWidth={0.5}
                fontSize={TEXT_SIZE}
                fontWeight="bold"
                textAnchor="middle"
              >
                {annotation.text}
              </SvgText>
            </React.Fragment>
          );
        })}
      </Svg>

      {/* Phase label */}
      {swingPhase && (
        <View style={styles.phaseContainer}>
          <Text style={styles.phaseText}>
            {swingPhase.replace('-', ' ').toUpperCase()}
          </Text>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    position: 'relative',
    backgroundColor: '#000',
    overflow: 'hidden',
    borderRadius: 8,
  },
  phaseContainer: {
    position: 'absolute',
    top: 10,
    right: 10,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 15,
  },
  phaseText: {
    color: '#FFF',
    fontWeight: 'bold',
    fontSize: 12,
  },
});

export default PoseVisualizer;