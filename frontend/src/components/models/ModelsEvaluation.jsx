import React, { useState, useMemo, useRef } from 'react';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Stack,
  List,
  Box,
  Tabs,
  Tab,
  ListItem,
  Alert,
  ListSubheader,
  ListItemText,
} from '@mui/material';
import Plotly from 'plotly.js';

import { IndicatorPlot, HistogramPlot, ScatterPlot, BarPlot } from '../plots';
import { ItemListView } from '../items';
import TabPanel from '../TabPanel';
import { CategoryFilter } from '../filters';

const scatterData = [
  { x: 0.379, y: 0.289, z: 0.529, c: 7 },
  { x: 0.43, y: 0.164, z: 0.429, c: 10 },
  { x: 0.703, y: 0.211, z: 0.062, c: 5 },
  { x: 0.738, y: 0.91, z: 0.28, c: 10 },
  { x: 0.086, y: 0.651, z: 0.991, c: 1 },
  { x: 0.466, y: 0.008, z: 0.398, c: 7 },
  { x: 0.371, y: 0.373, z: 0.55, c: 1 },
  { x: 0.661, y: 0.644, z: 0.443, c: 7 },
  { x: 0.329, y: 0.508, z: 0.897, c: 9 },
  { x: 0.979, y: 0.72, z: 0.858, c: 9 },
  { x: 0.781, y: 0.529, z: 0.049, c: 10 },
  { x: 0.955, y: 0.361, z: 0.667, c: 1 },
  { x: 0.095, y: 0.111, z: 0.249, c: 4 },
  { x: 0.287, y: 0.101, z: 0.801, c: 3 },
  { x: 0.048, y: 0.508, z: 0.531, c: 2 },
  { x: 0.14, y: 0.216, z: 0.693, c: 9 },
  { x: 0.587, y: 0.692, z: 0.075, c: 6 },
  { x: 0.958, y: 0.609, z: 0.211, c: 1 },
  { x: 0.017, y: 0.382, z: 0.332, c: 9 },
  { x: 0.945, y: 0.634, z: 0.946, c: 10 },
  { x: 0.177, y: 0.756, z: 0.155, c: 10 },
  { x: 0.773, y: 0.053, z: 0.799, c: 3 },
  { x: 0.144, y: 0.276, z: 0.08, c: 3 },
  { x: 0.166, y: 0.125, z: 0.748, c: 2 },
  { x: 0.068, y: 0.246, z: 0.809, c: 8 },
  { x: 0.094, y: 0.807, z: 0.867, c: 9 },
  { x: 0.795, y: 0.818, z: 0.378, c: 6 },
  { x: 0.973, y: 0.21, z: 0.149, c: 6 },
  { x: 0.245, y: 0.55, z: 0.981, c: 4 },
  { x: 0.173, y: 0.867, z: 0.65, c: 9 },
  { x: 0.372, y: 0.514, z: 0.765, c: 5 },
  { x: 0.795, y: 0.934, z: 0.391, c: 2 },
  { x: 0.498, y: 0.917, z: 0.138, c: 8 },
  { x: 0.027, y: 0.232, z: 0.61, c: 1 },
  { x: 0.673, y: 0.244, z: 0.184, c: 8 },
  { x: 0.427, y: 0.217, z: 0.321, c: 4 },
  { x: 0.639, y: 0.847, z: 0.195, c: 1 },
  { x: 0.598, y: 0.337, z: 0.19, c: 8 },
  { x: 0.872, y: 0.415, z: 0.553, c: 7 },
  { x: 0.104, y: 0.693, z: 0.991, c: 2 },
  { x: 0.328, y: 0.772, z: 0.704, c: 8 },
  { x: 0.676, y: 0.2, z: 0.21, c: 4 },
  { x: 0.126, y: 0.915, z: 0.232, c: 2 },
  { x: 0.9, y: 0.564, z: 0.281, c: 3 },
  { x: 0.178, y: 0.33, z: 0.096, c: 1 },
  { x: 0.541, y: 0.324, z: 0.154, c: 3 },
  { x: 0.31, y: 0.029, z: 0.873, c: 6 },
  { x: 0.045, y: 0.79, z: 0.608, c: 4 },
  { x: 0.266, y: 0.014, z: 0.224, c: 5 },
  { x: 0.959, y: 0.077, z: 0.708, c: 4 },
  { x: 0.479, y: 0.644, z: 0.505, c: 1 },
  { x: 0.334, y: 0.322, z: 0.859, c: 1 },
  { x: 0.104, y: 0.931, z: 0.031, c: 10 },
  { x: 0.136, y: 0.156, z: 0.453, c: 5 },
  { x: 0.59, y: 0.478, z: 0.686, c: 10 },
  { x: 0.225, y: 0.954, z: 0.021, c: 7 },
  { x: 0.328, y: 0.689, z: 0.466, c: 9 },
  { x: 0.546, y: 0.826, z: 0.788, c: 1 },
  { x: 0.508, y: 0.594, z: 0.652, c: 8 },
  { x: 0.505, y: 0.487, z: 0.759, c: 9 },
  { x: 0.67, y: 0.286, z: 0.19, c: 9 },
  { x: 0.52, y: 0.05, z: 0.781, c: 7 },
  { x: 0.232, y: 0.756, z: 0.669, c: 1 },
  { x: 0.347, y: 0.772, z: 0.353, c: 3 },
  { x: 0.157, y: 0.544, z: 0.512, c: 6 },
  { x: 0.042, y: 0.081, z: 0.385, c: 2 },
  { x: 0.139, y: 0.635, z: 0.484, c: 4 },
  { x: 0.816, y: 0.231, z: 0.352, c: 3 },
  { x: 0.427, y: 0.073, z: 0.227, c: 4 },
  { x: 0.593, y: 0.527, z: 0.304, c: 4 },
  { x: 0.757, y: 0.839, z: 0.219, c: 7 },
  { x: 0.451, y: 0.913, z: 0.362, c: 4 },
  { x: 0.897, y: 0.26, z: 0.865, c: 10 },
  { x: 0.708, y: 0.066, z: 0.11, c: 5 },
  { x: 0.146, y: 0.01, z: 0.465, c: 2 },
  { x: 0.459, y: 0.668, z: 0.136, c: 6 },
  { x: 0.776, y: 0.807, z: 0.841, c: 4 },
  { x: 0.192, y: 0.417, z: 0.405, c: 1 },
  { x: 0.449, y: 0.329, z: 0.134, c: 6 },
  { x: 0.046, y: 0.869, z: 0.692, c: 6 },
  { x: 0.999, y: 0.74, z: 0.319, c: 1 },
  { x: 0.064, y: 0.807, z: 0.813, c: 2 },
  { x: 0.337, y: 0.349, z: 0.614, c: 8 },
  { x: 0.962, y: 0.005, z: 0.028, c: 6 },
  { x: 0.747, y: 0.111, z: 0.674, c: 2 },
  { x: 0.456, y: 0.945, z: 0.24, c: 10 },
  { x: 0.442, y: 0.324, z: 0.132, c: 4 },
  { x: 0.041, y: 0.858, z: 0.062, c: 1 },
  { x: 0.196, y: 0.817, z: 0.696, c: 1 },
  { x: 0.383, y: 0.351, z: 0.607, c: 4 },
  { x: 0.906, y: 0.904, z: 0.268, c: 1 },
  { x: 0.062, y: 0.036, z: 0.244, c: 1 },
  { x: 0.813, y: 0.57, z: 0.888, c: 6 },
  { x: 1.0, y: 0.624, z: 0.429, c: 10 },
  { x: 0.334, y: 0.707, z: 0.825, c: 5 },
  { x: 0.282, y: 0.591, z: 0.207, c: 10 },
  { x: 0.072, y: 0.643, z: 0.248, c: 2 },
  { x: 0.372, y: 0.031, z: 0.498, c: 1 },
  { x: 0.963, y: 0.546, z: 0.5, c: 5 },
  { x: 0.705, y: 0.285, z: 0.659, c: 7 },
];

const histData = {
  KNN: [
    { 'Recall@20': 0.298, 'Recall@50': 0.895, 'Recall@100': 0.877 },
    { 'Recall@20': 0.278, 'Recall@50': 0.1, 'Recall@100': 0.0 },
    { 'Recall@20': 0.617, 'Recall@50': 0.382, 'Recall@100': 0.886 },
    { 'Recall@20': 0.204, 'Recall@50': 0.755, 'Recall@100': 0.888 },
    { 'Recall@20': 0.328, 'Recall@50': 0.854, 'Recall@100': 0.801 },
    { 'Recall@20': 0.474, 'Recall@50': 0.893, 'Recall@100': 0.78 },
    { 'Recall@20': 0.602, 'Recall@50': 0.668, 'Recall@100': 0.908 },
    { 'Recall@20': 0.238, 'Recall@50': 0.276, 'Recall@100': 0.211 },
    { 'Recall@20': 0.22, 'Recall@50': 0.951, 'Recall@100': 0.589 },
    { 'Recall@20': 0.273, 'Recall@50': 0.269, 'Recall@100': 0.009 },
    { 'Recall@20': 0.718, 'Recall@50': 0.265, 'Recall@100': 0.572 },
    { 'Recall@20': 0.653, 'Recall@50': 0.383, 'Recall@100': 0.923 },
    { 'Recall@20': 0.469, 'Recall@50': 0.791, 'Recall@100': 0.822 },
    { 'Recall@20': 0.307, 'Recall@50': 0.197, 'Recall@100': 0.743 },
    { 'Recall@20': 0.608, 'Recall@50': 0.624, 'Recall@100': 0.806 },
    { 'Recall@20': 0.668, 'Recall@50': 0.529, 'Recall@100': 0.045 },
    { 'Recall@20': 0.631, 'Recall@50': 0.586, 'Recall@100': 0.287 },
    { 'Recall@20': 0.113, 'Recall@50': 0.999, 'Recall@100': 0.956 },
    { 'Recall@20': 0.911, 'Recall@50': 0.125, 'Recall@100': 0.285 },
    { 'Recall@20': 0.902, 'Recall@50': 0.13, 'Recall@100': 0.617 },
    { 'Recall@20': 0.027, 'Recall@50': 0.403, 'Recall@100': 0.397 },
    { 'Recall@20': 0.876, 'Recall@50': 0.161, 'Recall@100': 0.722 },
    { 'Recall@20': 0.034, 'Recall@50': 0.428, 'Recall@100': 0.216 },
    { 'Recall@20': 0.74, 'Recall@50': 0.797, 'Recall@100': 0.381 },
    { 'Recall@20': 0.507, 'Recall@50': 0.635, 'Recall@100': 0.734 },
    { 'Recall@20': 0.719, 'Recall@50': 0.826, 'Recall@100': 0.28 },
    { 'Recall@20': 0.315, 'Recall@50': 0.816, 'Recall@100': 0.408 },
    { 'Recall@20': 0.813, 'Recall@50': 0.151, 'Recall@100': 0.852 },
    { 'Recall@20': 0.618, 'Recall@50': 0.57, 'Recall@100': 0.51 },
    { 'Recall@20': 0.341, 'Recall@50': 0.15, 'Recall@100': 0.001 },
    { 'Recall@20': 0.412, 'Recall@50': 0.69, 'Recall@100': 0.931 },
    { 'Recall@20': 0.719, 'Recall@50': 0.382, 'Recall@100': 0.613 },
    { 'Recall@20': 0.566, 'Recall@50': 0.379, 'Recall@100': 0.817 },
    { 'Recall@20': 0.727, 'Recall@50': 0.832, 'Recall@100': 0.603 },
    { 'Recall@20': 0.048, 'Recall@50': 0.099, 'Recall@100': 0.939 },
    { 'Recall@20': 0.352, 'Recall@50': 0.531, 'Recall@100': 0.921 },
    { 'Recall@20': 0.259, 'Recall@50': 0.187, 'Recall@100': 0.233 },
    { 'Recall@20': 0.781, 'Recall@50': 0.238, 'Recall@100': 0.36 },
    { 'Recall@20': 0.931, 'Recall@50': 0.533, 'Recall@100': 0.786 },
    { 'Recall@20': 0.864, 'Recall@50': 0.828, 'Recall@100': 0.438 },
    { 'Recall@20': 0.432, 'Recall@50': 0.296, 'Recall@100': 0.679 },
    { 'Recall@20': 0.497, 'Recall@50': 0.349, 'Recall@100': 0.35 },
    { 'Recall@20': 0.614, 'Recall@50': 0.577, 'Recall@100': 0.474 },
    { 'Recall@20': 0.095, 'Recall@50': 0.051, 'Recall@100': 0.089 },
    { 'Recall@20': 0.542, 'Recall@50': 0.884, 'Recall@100': 0.116 },
    { 'Recall@20': 0.303, 'Recall@50': 0.263, 'Recall@100': 0.211 },
    { 'Recall@20': 0.153, 'Recall@50': 0.95, 'Recall@100': 0.561 },
    { 'Recall@20': 0.371, 'Recall@50': 0.628, 'Recall@100': 0.516 },
    { 'Recall@20': 0.808, 'Recall@50': 0.661, 'Recall@100': 0.374 },
    { 'Recall@20': 0.7, 'Recall@50': 0.866, 'Recall@100': 0.383 },
    { 'Recall@20': 0.255, 'Recall@50': 0.426, 'Recall@100': 0.586 },
    { 'Recall@20': 0.149, 'Recall@50': 0.905, 'Recall@100': 0.145 },
    { 'Recall@20': 0.127, 'Recall@50': 0.574, 'Recall@100': 0.16 },
    { 'Recall@20': 0.761, 'Recall@50': 0.3, 'Recall@100': 0.08 },
    { 'Recall@20': 0.506, 'Recall@50': 0.089, 'Recall@100': 0.817 },
    { 'Recall@20': 0.506, 'Recall@50': 0.137, 'Recall@100': 0.319 },
    { 'Recall@20': 0.241, 'Recall@50': 0.347, 'Recall@100': 0.765 },
    { 'Recall@20': 0.364, 'Recall@50': 0.609, 'Recall@100': 0.161 },
    { 'Recall@20': 0.46, 'Recall@50': 0.389, 'Recall@100': 0.35 },
    { 'Recall@20': 0.224, 'Recall@50': 0.71, 'Recall@100': 0.448 },
    { 'Recall@20': 0.088, 'Recall@50': 0.791, 'Recall@100': 0.07 },
    { 'Recall@20': 0.323, 'Recall@50': 0.655, 'Recall@100': 0.586 },
    { 'Recall@20': 0.253, 'Recall@50': 0.922, 'Recall@100': 0.135 },
    { 'Recall@20': 0.112, 'Recall@50': 0.36, 'Recall@100': 0.753 },
    { 'Recall@20': 0.925, 'Recall@50': 0.908, 'Recall@100': 0.769 },
    { 'Recall@20': 0.544, 'Recall@50': 0.097, 'Recall@100': 0.528 },
    { 'Recall@20': 0.821, 'Recall@50': 0.856, 'Recall@100': 0.241 },
    { 'Recall@20': 0.156, 'Recall@50': 0.279, 'Recall@100': 0.322 },
    { 'Recall@20': 0.687, 'Recall@50': 0.638, 'Recall@100': 0.475 },
    { 'Recall@20': 0.611, 'Recall@50': 0.592, 'Recall@100': 0.643 },
    { 'Recall@20': 0.787, 'Recall@50': 0.039, 'Recall@100': 0.061 },
    { 'Recall@20': 0.247, 'Recall@50': 0.777, 'Recall@100': 0.154 },
    { 'Recall@20': 0.151, 'Recall@50': 0.148, 'Recall@100': 0.182 },
    { 'Recall@20': 0.651, 'Recall@50': 0.288, 'Recall@100': 0.71 },
    { 'Recall@20': 0.388, 'Recall@50': 0.033, 'Recall@100': 0.074 },
    { 'Recall@20': 0.716, 'Recall@50': 0.759, 'Recall@100': 0.871 },
    { 'Recall@20': 0.624, 'Recall@50': 0.173, 'Recall@100': 0.224 },
    { 'Recall@20': 0.148, 'Recall@50': 0.952, 'Recall@100': 0.348 },
    { 'Recall@20': 0.037, 'Recall@50': 0.039, 'Recall@100': 0.403 },
    { 'Recall@20': 0.175, 'Recall@50': 0.39, 'Recall@100': 0.642 },
    { 'Recall@20': 0.008, 'Recall@50': 0.026, 'Recall@100': 0.283 },
    { 'Recall@20': 0.9, 'Recall@50': 0.32, 'Recall@100': 0.437 },
    { 'Recall@20': 0.715, 'Recall@50': 0.92, 'Recall@100': 0.992 },
    { 'Recall@20': 0.91, 'Recall@50': 0.411, 'Recall@100': 0.989 },
    { 'Recall@20': 0.311, 'Recall@50': 0.082, 'Recall@100': 0.756 },
    { 'Recall@20': 0.782, 'Recall@50': 0.172, 'Recall@100': 0.845 },
    { 'Recall@20': 0.596, 'Recall@50': 0.828, 'Recall@100': 0.45 },
    { 'Recall@20': 0.584, 'Recall@50': 0.099, 'Recall@100': 0.979 },
    { 'Recall@20': 0.975, 'Recall@50': 0.176, 'Recall@100': 0.133 },
    { 'Recall@20': 0.467, 'Recall@50': 0.795, 'Recall@100': 0.469 },
    { 'Recall@20': 0.947, 'Recall@50': 0.214, 'Recall@100': 0.362 },
    { 'Recall@20': 0.681, 'Recall@50': 0.666, 'Recall@100': 0.295 },
    { 'Recall@20': 0.913, 'Recall@50': 0.157, 'Recall@100': 0.65 },
    { 'Recall@20': 0.656, 'Recall@50': 0.011, 'Recall@100': 0.221 },
    { 'Recall@20': 0.115, 'Recall@50': 0.968, 'Recall@100': 0.025 },
    { 'Recall@20': 0.716, 'Recall@50': 0.439, 'Recall@100': 0.22 },
    { 'Recall@20': 0.513, 'Recall@50': 0.468, 'Recall@100': 0.692 },
    { 'Recall@20': 0.789, 'Recall@50': 0.861, 'Recall@100': 0.252 },
    { 'Recall@20': 0.407, 'Recall@50': 0.35, 'Recall@100': 0.093 },
    { 'Recall@20': 0.422, 'Recall@50': 0.612, 'Recall@100': 0.39 },
  ],
  SVD: [
    { 'Recall@20': 0.691, 'Recall@50': 0.762, 'Recall@100': 0.005 },
    { 'Recall@20': 0.085, 'Recall@50': 0.528, 'Recall@100': 0.531 },
    { 'Recall@20': 0.362, 'Recall@50': 0.221, 'Recall@100': 0.951 },
    { 'Recall@20': 0.694, 'Recall@50': 0.147, 'Recall@100': 0.022 },
    { 'Recall@20': 0.995, 'Recall@50': 0.086, 'Recall@100': 0.578 },
    { 'Recall@20': 0.243, 'Recall@50': 0.787, 'Recall@100': 0.488 },
    { 'Recall@20': 0.84, 'Recall@50': 0.053, 'Recall@100': 0.313 },
    { 'Recall@20': 0.062, 'Recall@50': 0.264, 'Recall@100': 0.035 },
    { 'Recall@20': 0.551, 'Recall@50': 0.99, 'Recall@100': 0.323 },
    { 'Recall@20': 0.917, 'Recall@50': 0.894, 'Recall@100': 0.174 },
    { 'Recall@20': 0.762, 'Recall@50': 0.524, 'Recall@100': 0.105 },
    { 'Recall@20': 0.973, 'Recall@50': 0.713, 'Recall@100': 0.246 },
    { 'Recall@20': 0.637, 'Recall@50': 0.318, 'Recall@100': 0.661 },
    { 'Recall@20': 0.975, 'Recall@50': 0.151, 'Recall@100': 0.515 },
    { 'Recall@20': 0.733, 'Recall@50': 0.972, 'Recall@100': 0.051 },
    { 'Recall@20': 0.605, 'Recall@50': 0.273, 'Recall@100': 0.171 },
    { 'Recall@20': 0.774, 'Recall@50': 0.575, 'Recall@100': 0.3 },
    { 'Recall@20': 0.117, 'Recall@50': 0.579, 'Recall@100': 0.996 },
    { 'Recall@20': 0.568, 'Recall@50': 0.102, 'Recall@100': 0.349 },
    { 'Recall@20': 0.434, 'Recall@50': 0.575, 'Recall@100': 0.975 },
    { 'Recall@20': 0.232, 'Recall@50': 0.936, 'Recall@100': 0.548 },
    { 'Recall@20': 0.201, 'Recall@50': 0.65, 'Recall@100': 0.567 },
    { 'Recall@20': 0.154, 'Recall@50': 0.825, 'Recall@100': 0.85 },
    { 'Recall@20': 0.275, 'Recall@50': 0.924, 'Recall@100': 0.24 },
    { 'Recall@20': 0.556, 'Recall@50': 0.365, 'Recall@100': 0.737 },
    { 'Recall@20': 0.38, 'Recall@50': 0.4, 'Recall@100': 0.411 },
    { 'Recall@20': 0.832, 'Recall@50': 0.459, 'Recall@100': 0.344 },
    { 'Recall@20': 0.484, 'Recall@50': 0.754, 'Recall@100': 0.633 },
    { 'Recall@20': 0.056, 'Recall@50': 0.291, 'Recall@100': 0.656 },
    { 'Recall@20': 0.583, 'Recall@50': 0.12, 'Recall@100': 0.546 },
    { 'Recall@20': 0.49, 'Recall@50': 0.426, 'Recall@100': 0.986 },
    { 'Recall@20': 0.385, 'Recall@50': 0.989, 'Recall@100': 0.172 },
    { 'Recall@20': 0.138, 'Recall@50': 0.692, 'Recall@100': 0.432 },
    { 'Recall@20': 0.603, 'Recall@50': 0.795, 'Recall@100': 0.75 },
    { 'Recall@20': 0.266, 'Recall@50': 0.79, 'Recall@100': 0.504 },
    { 'Recall@20': 0.242, 'Recall@50': 0.051, 'Recall@100': 0.838 },
    { 'Recall@20': 0.35, 'Recall@50': 0.575, 'Recall@100': 0.55 },
    { 'Recall@20': 0.971, 'Recall@50': 0.696, 'Recall@100': 0.232 },
    { 'Recall@20': 0.099, 'Recall@50': 0.123, 'Recall@100': 0.081 },
    { 'Recall@20': 0.762, 'Recall@50': 0.674, 'Recall@100': 0.252 },
    { 'Recall@20': 0.05, 'Recall@50': 0.181, 'Recall@100': 0.507 },
    { 'Recall@20': 0.181, 'Recall@50': 0.045, 'Recall@100': 0.462 },
    { 'Recall@20': 0.33, 'Recall@50': 0.972, 'Recall@100': 0.073 },
    { 'Recall@20': 0.048, 'Recall@50': 0.484, 'Recall@100': 0.292 },
    { 'Recall@20': 0.018, 'Recall@50': 0.239, 'Recall@100': 0.826 },
    { 'Recall@20': 0.591, 'Recall@50': 0.309, 'Recall@100': 0.766 },
    { 'Recall@20': 0.361, 'Recall@50': 0.806, 'Recall@100': 0.276 },
    { 'Recall@20': 0.551, 'Recall@50': 0.512, 'Recall@100': 0.314 },
    { 'Recall@20': 0.388, 'Recall@50': 0.612, 'Recall@100': 0.971 },
    { 'Recall@20': 0.918, 'Recall@50': 0.963, 'Recall@100': 0.778 },
    { 'Recall@20': 0.833, 'Recall@50': 0.901, 'Recall@100': 0.145 },
    { 'Recall@20': 0.658, 'Recall@50': 0.654, 'Recall@100': 0.831 },
    { 'Recall@20': 0.357, 'Recall@50': 0.56, 'Recall@100': 0.6 },
    { 'Recall@20': 0.694, 'Recall@50': 0.002, 'Recall@100': 0.719 },
    { 'Recall@20': 0.239, 'Recall@50': 0.922, 'Recall@100': 0.118 },
    { 'Recall@20': 0.331, 'Recall@50': 0.667, 'Recall@100': 0.044 },
    { 'Recall@20': 0.452, 'Recall@50': 0.697, 'Recall@100': 0.315 },
    { 'Recall@20': 0.559, 'Recall@50': 0.87, 'Recall@100': 0.248 },
    { 'Recall@20': 0.152, 'Recall@50': 0.607, 'Recall@100': 0.345 },
    { 'Recall@20': 0.13, 'Recall@50': 0.324, 'Recall@100': 0.321 },
    { 'Recall@20': 0.558, 'Recall@50': 0.916, 'Recall@100': 0.228 },
    { 'Recall@20': 0.002, 'Recall@50': 0.428, 'Recall@100': 0.584 },
    { 'Recall@20': 0.121, 'Recall@50': 0.625, 'Recall@100': 0.507 },
    { 'Recall@20': 0.117, 'Recall@50': 0.505, 'Recall@100': 0.503 },
    { 'Recall@20': 0.676, 'Recall@50': 0.31, 'Recall@100': 0.448 },
    { 'Recall@20': 0.49, 'Recall@50': 0.283, 'Recall@100': 0.106 },
    { 'Recall@20': 0.271, 'Recall@50': 0.9, 'Recall@100': 0.088 },
    { 'Recall@20': 0.252, 'Recall@50': 0.274, 'Recall@100': 0.662 },
    { 'Recall@20': 0.479, 'Recall@50': 0.453, 'Recall@100': 0.802 },
    { 'Recall@20': 0.682, 'Recall@50': 0.383, 'Recall@100': 0.395 },
    { 'Recall@20': 0.13, 'Recall@50': 0.931, 'Recall@100': 0.833 },
    { 'Recall@20': 0.396, 'Recall@50': 0.351, 'Recall@100': 0.526 },
    { 'Recall@20': 0.159, 'Recall@50': 0.861, 'Recall@100': 0.717 },
    { 'Recall@20': 0.063, 'Recall@50': 0.018, 'Recall@100': 0.522 },
    { 'Recall@20': 0.689, 'Recall@50': 0.791, 'Recall@100': 0.792 },
    { 'Recall@20': 0.184, 'Recall@50': 0.314, 'Recall@100': 0.374 },
    { 'Recall@20': 0.491, 'Recall@50': 0.709, 'Recall@100': 0.942 },
    { 'Recall@20': 0.444, 'Recall@50': 0.299, 'Recall@100': 0.795 },
    { 'Recall@20': 0.06, 'Recall@50': 0.449, 'Recall@100': 0.001 },
    { 'Recall@20': 0.794, 'Recall@50': 0.967, 'Recall@100': 0.115 },
    { 'Recall@20': 0.135, 'Recall@50': 0.967, 'Recall@100': 0.121 },
    { 'Recall@20': 0.766, 'Recall@50': 0.089, 'Recall@100': 0.028 },
    { 'Recall@20': 0.119, 'Recall@50': 0.097, 'Recall@100': 0.96 },
    { 'Recall@20': 0.055, 'Recall@50': 0.203, 'Recall@100': 0.866 },
    { 'Recall@20': 0.052, 'Recall@50': 0.814, 'Recall@100': 0.516 },
    { 'Recall@20': 0.917, 'Recall@50': 0.829, 'Recall@100': 0.081 },
    { 'Recall@20': 0.614, 'Recall@50': 0.609, 'Recall@100': 0.753 },
    { 'Recall@20': 0.851, 'Recall@50': 0.733, 'Recall@100': 0.919 },
    { 'Recall@20': 0.261, 'Recall@50': 0.445, 'Recall@100': 0.609 },
    { 'Recall@20': 0.133, 'Recall@50': 0.862, 'Recall@100': 0.585 },
    { 'Recall@20': 0.495, 'Recall@50': 0.676, 'Recall@100': 0.156 },
    { 'Recall@20': 0.921, 'Recall@50': 0.813, 'Recall@100': 0.477 },
    { 'Recall@20': 0.682, 'Recall@50': 0.664, 'Recall@100': 0.422 },
    { 'Recall@20': 0.226, 'Recall@50': 0.3, 'Recall@100': 0.38 },
    { 'Recall@20': 0.081, 'Recall@50': 0.971, 'Recall@100': 0.65 },
    { 'Recall@20': 0.062, 'Recall@50': 0.066, 'Recall@100': 0.418 },
    { 'Recall@20': 0.572, 'Recall@50': 0.387, 'Recall@100': 0.352 },
    { 'Recall@20': 0.893, 'Recall@50': 0.183, 'Recall@100': 0.31 },
    { 'Recall@20': 0.214, 'Recall@50': 0.706, 'Recall@100': 0.474 },
    { 'Recall@20': 0.878, 'Recall@50': 0.458, 'Recall@100': 0.925 },
  ],
};

const summaryData = [
  {
    name: 'KNN',
    metrics: {
      'Recall@20': 0.3,
      'Recall@50': 0.4,
      'NDCG@100': 0.1,
      'Coverage@20': 0.3,
      'Coverage@50': 0.8,
      'Coverage@100': 0.2,
      'Novelty@10': 0.65,
      'Recall@100': 0.32,
    },
    metricsPrev: {
      'Recall@20': 0.2,
      'Recall@50': 0.5,
      'NDCG@100': 0.3,
      'Coverage@20': 0.1,
      'Coverage@50': 0.7,
      'Coverage@100': 0.15,
    },
  },
  {
    name: 'SVD',
    metrics: {
      'Recall@20': 0.2,
      'Recall@50': 0.5,
      'NDCG@100': 0.23,
      'Coverage@20': 0.36,
      'Coverage@50': 0.78,
      'Coverage@100': 0.23,
      'Novelty@10': 0.32,
      'Recall@100': 0.42,
    },
  },
  {
    name: 'VASP',
    metrics: {
      'Recall@20': 0.2,
      'Recall@50': 0.5,
      'NDCG@100': 0.23,
      'Coverage@20': 0.36,
      'Coverage@50': 0.78,
      'Coverage@100': 0.23,
      'Novelty@10': 0.43,
      'Recall@100': 0.12,
    },
  },
];

const activeColor = '#ef553b';
const defaultColor = '#DCDCDC';

function ModelsEvaluation() {
  const [selectedModel, setSelectedModel] = useState(Object.keys(histData)[0]);
  const [selectedMetric, setSelectedMetric] = useState(Object.keys(histData[selectedModel][0])[0]);
  const [selectedUsers, setSelectedUsers] = useState([]);

  const [histTab, setHistTab] = useState(0);
  const [modelTab, setModelTab] = useState(0);

  const histRef = useRef();

  const handleHistTabChange = (event, newValue) => {
    setHistTab(newValue);
  };

  const handleModelTabChange = (e, modelIndex) => {
    setModelTab(modelIndex);
  };

  const resetHistSelection = () => {
    setSelectedUsers([]);
    Plotly.restyle(histRef.current.el, { selectedpoints: [null] });
  };

  const scatterColors = useMemo(() => {
    if (selectedUsers.length === 0) {
      return [];
    }
    const colors = [];
    for (let i = 0; i < 100; i += 1) colors.push(defaultColor);
    selectedUsers.forEach((p) => {
      colors[p] = activeColor;
    });
    return colors;
  }, [selectedUsers]);

  const scatterPoints = useMemo(
    () => ({
      x: scatterData.map(({ x }) => x),
      y: scatterData.map(({ y }) => y),
      z: scatterData.map(({ z }) => z),
    }),
    []
  );

  return (
    <Container maxWidth="xl">
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Box pl={1}>
            <Typography component="div" variant="h6">
              Models Performance
            </Typography>
            <Typography variant="subtitle1" gutterBottom>
              A performance in the individual metrics with comparasion to the previous evaluation
            </Typography>
          </Box>

          <Grid container spacing={2}>
            <Grid item xs={7}>
              <Paper sx={{ height: '100%' }}>
                <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                  <Tabs value={modelTab} onChange={handleModelTabChange} variant="fullWidth">
                    {summaryData.map((m) => (
                      <Tab label={m.name} key={m.name} />
                    ))}
                  </Tabs>
                </Box>
                <Box sx={{ p: 2 }}>
                  <Grid container>
                    {Object.entries(summaryData[modelTab].metrics).map(([metric, value]) => (
                      <Grid item xs={3} key={metric}>
                        <IndicatorPlot
                          title={metric}
                          height={150}
                          value={value}
                          delta={
                            summaryData[0].metricsPrev && summaryData[0].metricsPrev[metric]
                              ? summaryData[0].metricsPrev[metric]
                              : 0
                          }
                        />
                      </Grid>
                    ))}
                  </Grid>
                </Box>
              </Paper>
            </Grid>
            {summaryData.length > 1 && (
              <Grid item xs={5}>
                <Paper>
                  <BarPlot
                    orientation="h"
                    data={summaryData.map((model) => ({
                      y: Object.keys(model.metrics),
                      x: Object.values(model.metrics),
                      name: model.name,
                    }))}
                    layoutProps={{
                      margin: { t: 30, b: 40, l: 120, r: 40 },
                    }}
                    height={400}
                  />
                </Paper>
              </Grid>
            )}
          </Grid>
        </Grid>
        <Grid item xs={12}>
          <Box pl={1}>
            <Typography component="div" variant="h6">
              Metrics Distribution
            </Typography>
            <Typography variant="subtitle1" gutterBottom>
              A distribution of the metrics for each validation user
            </Typography>
          </Box>
          <Grid container spacing={2}>
            <Grid item xs={7}>
              <Paper sx={{ p: 2 }}>
                <Stack direction="row" spacing={2}>
                  <CategoryFilter
                    label="Model"
                    value={selectedModel}
                    onChange={(newValue) => {
                      setSelectedModel(newValue);
                      setSelectedMetric(Object.keys(histData[selectedModel][0])[0]);
                      resetHistSelection();
                    }}
                    options={Object.keys(histData).map((model) => ({
                      value: model,
                      label: model,
                    }))}
                  />
                  <CategoryFilter
                    label="Metric"
                    value={selectedMetric}
                    onChange={(newValue) => {
                      setSelectedMetric(newValue);
                      resetHistSelection();
                    }}
                    options={Object.keys(histData[selectedModel][0]).map((metric) => ({
                      value: metric,
                      label: metric,
                    }))}
                  />
                </Stack>
                <HistogramPlot
                  data={histData[selectedModel].map((d) => d[selectedMetric])}
                  height={350}
                  innerRef={histRef}
                  onDeselect={() => {
                    setSelectedUsers([]);
                  }}
                  onSelected={(eventData) => {
                    if (eventData) {
                      const { points } = eventData;
                      const { selectedpoints } = points[0].data;
                      setSelectedUsers(selectedpoints);
                    }
                  }}
                />
              </Paper>
            </Grid>
            <Grid item xs={5}>
              <Paper sx={{ height: '100%' }}>
                <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                  <Tabs value={histTab} onChange={handleHistTabChange} variant="fullWidth">
                    <Tab label="User embeddings" />
                    <Tab label="User details" />
                  </Tabs>
                </Box>
                {selectedUsers.length > 0 ? (
                  <>
                    <TabPanel value={histTab} index={0}>
                      <ScatterPlot
                        height={380}
                        x={scatterPoints.x}
                        y={scatterPoints.y}
                        z={scatterPoints.z}
                        color={scatterColors}
                      />
                    </TabPanel>
                    <TabPanel value={histTab} index={1}>
                      <Grid container spacing={1} sx={{ p: 1 }}>
                        <Grid item xs={5}>
                          <List
                            dense
                            subheader={
                              <ListSubheader component="div">Users Characteristic</ListSubheader>
                            }
                          >
                            <ListItem>
                              <ListItemText
                                primary="45% users"
                                secondary="Ratio of the selected users"
                              />
                            </ListItem>
                            <ListItem>
                              <ListItemText
                                primary="30% similarity"
                                secondary="Similarity between users"
                              />
                            </ListItem>
                            <ListItem>
                              <ListItemText
                                primary="15.6 interactions"
                                secondary="Avg. number of interactions"
                              />
                            </ListItem>
                            <ListItem>
                              <ListItemText
                                primary="14% conformity"
                                secondary="Ratio of items agreement"
                              />
                            </ListItem>
                          </List>
                        </Grid>
                        <Grid item xs={7}>
                          <List
                            dense
                            subheader={
                              <ListSubheader component="div">
                                The Most Interacted Items
                              </ListSubheader>
                            }
                          >
                            <ItemListView
                              title="Four Weddings and a Funeral (1994)"
                              subtitle="Comedy, Drama, Romance"
                              image="https://m.media-amazon.com/images/M/MV5BMTMyNzg2NzgxNV5BMl5BanBnXkFtZTcwMTcxNzczNA@@..jpg"
                            />
                            <ItemListView
                              title="Cutthroat Island (1995)"
                              subtitle="Action, Adventure, Comedy"
                            />
                            <ItemListView
                              title="Four Weddings and a Funeral (1994)"
                              subtitle="Comedy, Drama, Romance"
                              image="https://m.media-amazon.com/images/M/MV5BMTMyNzg2NzgxNV5BMl5BanBnXkFtZTcwMTcxNzczNA@@..jpg"
                            />
                            <ItemListView
                              title="Cutthroat Island (1995)"
                              subtitle="Action, Adventure, Comedy"
                            />
                            <ItemListView
                              title="Cutthroat Island (1995)"
                              subtitle="Action, Adventure, Comedy"
                            />
                          </List>
                        </Grid>
                      </Grid>
                    </TabPanel>
                  </>
                ) : (
                  <Box sx={{ p: 2 }}>
                    <Alert severity="info">
                      To see the details, select a range of users in the histogram plot.
                    </Alert>
                  </Box>
                )}
              </Paper>
            </Grid>
          </Grid>
        </Grid>
      </Grid>
    </Container>
  );
}

export default ModelsEvaluation;