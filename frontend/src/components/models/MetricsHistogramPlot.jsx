import React, { useState, useRef, useMemo, useEffect } from 'react';
import pt from 'prop-types';
import { Paper, Stack, Box } from '@mui/material';
import Plotly from 'plotly.js';

import { CategoryFilter } from '../filters';
import { HistogramPlot } from '../plots';

const histogramData = {
  KNN: [
    { id: 0, 'Recall@20': 0.298, 'Recall@50': 0.895, 'Recall@100': 0.877 },
    { id: 0, 'Recall@20': 0.278, 'Recall@50': 0.1, 'Recall@100': 0.0 },
    { id: 0, 'Recall@20': 0.617, 'Recall@50': 0.382, 'Recall@100': 0.886 },
    { id: 0, 'Recall@20': 0.204, 'Recall@50': 0.755, 'Recall@100': 0.888 },
    { id: 0, 'Recall@20': 0.328, 'Recall@50': 0.854, 'Recall@100': 0.801 },
    { id: 0, 'Recall@20': 0.474, 'Recall@50': 0.893, 'Recall@100': 0.78 },
    { id: 0, 'Recall@20': 0.602, 'Recall@50': 0.668, 'Recall@100': 0.908 },
    { id: 0, 'Recall@20': 0.238, 'Recall@50': 0.276, 'Recall@100': 0.211 },
    { id: 0, 'Recall@20': 0.22, 'Recall@50': 0.951, 'Recall@100': 0.589 },
    { id: 0, 'Recall@20': 0.273, 'Recall@50': 0.269, 'Recall@100': 0.009 },
    { id: 0, 'Recall@20': 0.718, 'Recall@50': 0.265, 'Recall@100': 0.572 },
    { id: 0, 'Recall@20': 0.653, 'Recall@50': 0.383, 'Recall@100': 0.923 },
    { id: 0, 'Recall@20': 0.469, 'Recall@50': 0.791, 'Recall@100': 0.822 },
    { id: 0, 'Recall@20': 0.307, 'Recall@50': 0.197, 'Recall@100': 0.743 },
    { id: 0, 'Recall@20': 0.608, 'Recall@50': 0.624, 'Recall@100': 0.806 },
    { id: 0, 'Recall@20': 0.668, 'Recall@50': 0.529, 'Recall@100': 0.045 },
    { id: 0, 'Recall@20': 0.631, 'Recall@50': 0.586, 'Recall@100': 0.287 },
    { id: 0, 'Recall@20': 0.113, 'Recall@50': 0.999, 'Recall@100': 0.956 },
    { id: 0, 'Recall@20': 0.911, 'Recall@50': 0.125, 'Recall@100': 0.285 },
    { id: 0, 'Recall@20': 0.902, 'Recall@50': 0.13, 'Recall@100': 0.617 },
    { id: 0, 'Recall@20': 0.027, 'Recall@50': 0.403, 'Recall@100': 0.397 },
    { id: 0, 'Recall@20': 0.876, 'Recall@50': 0.161, 'Recall@100': 0.722 },
    { id: 0, 'Recall@20': 0.034, 'Recall@50': 0.428, 'Recall@100': 0.216 },
    { id: 0, 'Recall@20': 0.74, 'Recall@50': 0.797, 'Recall@100': 0.381 },
    { id: 0, 'Recall@20': 0.507, 'Recall@50': 0.635, 'Recall@100': 0.734 },
    { id: 0, 'Recall@20': 0.719, 'Recall@50': 0.826, 'Recall@100': 0.28 },
    { id: 0, 'Recall@20': 0.315, 'Recall@50': 0.816, 'Recall@100': 0.408 },
    { id: 0, 'Recall@20': 0.813, 'Recall@50': 0.151, 'Recall@100': 0.852 },
    { id: 0, 'Recall@20': 0.618, 'Recall@50': 0.57, 'Recall@100': 0.51 },
    { id: 0, 'Recall@20': 0.341, 'Recall@50': 0.15, 'Recall@100': 0.001 },
    { id: 0, 'Recall@20': 0.412, 'Recall@50': 0.69, 'Recall@100': 0.931 },
    { id: 0, 'Recall@20': 0.719, 'Recall@50': 0.382, 'Recall@100': 0.613 },
    { id: 0, 'Recall@20': 0.566, 'Recall@50': 0.379, 'Recall@100': 0.817 },
    { id: 0, 'Recall@20': 0.727, 'Recall@50': 0.832, 'Recall@100': 0.603 },
    { id: 0, 'Recall@20': 0.048, 'Recall@50': 0.099, 'Recall@100': 0.939 },
    { id: 0, 'Recall@20': 0.352, 'Recall@50': 0.531, 'Recall@100': 0.921 },
    { id: 0, 'Recall@20': 0.259, 'Recall@50': 0.187, 'Recall@100': 0.233 },
    { id: 0, 'Recall@20': 0.781, 'Recall@50': 0.238, 'Recall@100': 0.36 },
    { id: 0, 'Recall@20': 0.931, 'Recall@50': 0.533, 'Recall@100': 0.786 },
    { id: 0, 'Recall@20': 0.864, 'Recall@50': 0.828, 'Recall@100': 0.438 },
    { id: 0, 'Recall@20': 0.432, 'Recall@50': 0.296, 'Recall@100': 0.679 },
    { id: 0, 'Recall@20': 0.497, 'Recall@50': 0.349, 'Recall@100': 0.35 },
    { id: 0, 'Recall@20': 0.614, 'Recall@50': 0.577, 'Recall@100': 0.474 },
    { id: 0, 'Recall@20': 0.095, 'Recall@50': 0.051, 'Recall@100': 0.089 },
    { id: 0, 'Recall@20': 0.542, 'Recall@50': 0.884, 'Recall@100': 0.116 },
    { id: 0, 'Recall@20': 0.303, 'Recall@50': 0.263, 'Recall@100': 0.211 },
    { id: 0, 'Recall@20': 0.153, 'Recall@50': 0.95, 'Recall@100': 0.561 },
    { id: 0, 'Recall@20': 0.371, 'Recall@50': 0.628, 'Recall@100': 0.516 },
    { id: 0, 'Recall@20': 0.808, 'Recall@50': 0.661, 'Recall@100': 0.374 },
    { id: 0, 'Recall@20': 0.7, 'Recall@50': 0.866, 'Recall@100': 0.383 },
    { id: 0, 'Recall@20': 0.255, 'Recall@50': 0.426, 'Recall@100': 0.586 },
    { id: 0, 'Recall@20': 0.149, 'Recall@50': 0.905, 'Recall@100': 0.145 },
    { id: 0, 'Recall@20': 0.127, 'Recall@50': 0.574, 'Recall@100': 0.16 },
    { id: 0, 'Recall@20': 0.761, 'Recall@50': 0.3, 'Recall@100': 0.08 },
    { id: 0, 'Recall@20': 0.506, 'Recall@50': 0.089, 'Recall@100': 0.817 },
    { id: 0, 'Recall@20': 0.506, 'Recall@50': 0.137, 'Recall@100': 0.319 },
    { id: 0, 'Recall@20': 0.241, 'Recall@50': 0.347, 'Recall@100': 0.765 },
    { id: 0, 'Recall@20': 0.364, 'Recall@50': 0.609, 'Recall@100': 0.161 },
    { id: 0, 'Recall@20': 0.46, 'Recall@50': 0.389, 'Recall@100': 0.35 },
    { id: 0, 'Recall@20': 0.224, 'Recall@50': 0.71, 'Recall@100': 0.448 },
    { id: 0, 'Recall@20': 0.088, 'Recall@50': 0.791, 'Recall@100': 0.07 },
    { id: 0, 'Recall@20': 0.323, 'Recall@50': 0.655, 'Recall@100': 0.586 },
    { id: 0, 'Recall@20': 0.253, 'Recall@50': 0.922, 'Recall@100': 0.135 },
    { id: 0, 'Recall@20': 0.112, 'Recall@50': 0.36, 'Recall@100': 0.753 },
    { id: 0, 'Recall@20': 0.925, 'Recall@50': 0.908, 'Recall@100': 0.769 },
    { id: 0, 'Recall@20': 0.544, 'Recall@50': 0.097, 'Recall@100': 0.528 },
    { id: 0, 'Recall@20': 0.821, 'Recall@50': 0.856, 'Recall@100': 0.241 },
    { id: 0, 'Recall@20': 0.156, 'Recall@50': 0.279, 'Recall@100': 0.322 },
    { id: 0, 'Recall@20': 0.687, 'Recall@50': 0.638, 'Recall@100': 0.475 },
    { id: 0, 'Recall@20': 0.611, 'Recall@50': 0.592, 'Recall@100': 0.643 },
    { id: 0, 'Recall@20': 0.787, 'Recall@50': 0.039, 'Recall@100': 0.061 },
    { id: 0, 'Recall@20': 0.247, 'Recall@50': 0.777, 'Recall@100': 0.154 },
    { id: 0, 'Recall@20': 0.151, 'Recall@50': 0.148, 'Recall@100': 0.182 },
    { id: 0, 'Recall@20': 0.651, 'Recall@50': 0.288, 'Recall@100': 0.71 },
    { id: 0, 'Recall@20': 0.388, 'Recall@50': 0.033, 'Recall@100': 0.074 },
    { id: 0, 'Recall@20': 0.716, 'Recall@50': 0.759, 'Recall@100': 0.871 },
    { id: 0, 'Recall@20': 0.624, 'Recall@50': 0.173, 'Recall@100': 0.224 },
    { id: 0, 'Recall@20': 0.148, 'Recall@50': 0.952, 'Recall@100': 0.348 },
    { id: 0, 'Recall@20': 0.037, 'Recall@50': 0.039, 'Recall@100': 0.403 },
    { id: 0, 'Recall@20': 0.175, 'Recall@50': 0.39, 'Recall@100': 0.642 },
    { id: 0, 'Recall@20': 0.008, 'Recall@50': 0.026, 'Recall@100': 0.283 },
    { id: 0, 'Recall@20': 0.9, 'Recall@50': 0.32, 'Recall@100': 0.437 },
    { id: 0, 'Recall@20': 0.715, 'Recall@50': 0.92, 'Recall@100': 0.992 },
    { id: 0, 'Recall@20': 0.91, 'Recall@50': 0.411, 'Recall@100': 0.989 },
    { id: 0, 'Recall@20': 0.311, 'Recall@50': 0.082, 'Recall@100': 0.756 },
    { id: 0, 'Recall@20': 0.782, 'Recall@50': 0.172, 'Recall@100': 0.845 },
    { id: 0, 'Recall@20': 0.596, 'Recall@50': 0.828, 'Recall@100': 0.45 },
    { id: 0, 'Recall@20': 0.584, 'Recall@50': 0.099, 'Recall@100': 0.979 },
    { id: 0, 'Recall@20': 0.975, 'Recall@50': 0.176, 'Recall@100': 0.133 },
    { id: 0, 'Recall@20': 0.467, 'Recall@50': 0.795, 'Recall@100': 0.469 },
    { id: 0, 'Recall@20': 0.947, 'Recall@50': 0.214, 'Recall@100': 0.362 },
    { id: 0, 'Recall@20': 0.681, 'Recall@50': 0.666, 'Recall@100': 0.295 },
    { id: 0, 'Recall@20': 0.913, 'Recall@50': 0.157, 'Recall@100': 0.65 },
    { id: 0, 'Recall@20': 0.656, 'Recall@50': 0.011, 'Recall@100': 0.221 },
    { id: 0, 'Recall@20': 0.115, 'Recall@50': 0.968, 'Recall@100': 0.025 },
    { id: 0, 'Recall@20': 0.716, 'Recall@50': 0.439, 'Recall@100': 0.22 },
    { id: 0, 'Recall@20': 0.513, 'Recall@50': 0.468, 'Recall@100': 0.692 },
    { id: 0, 'Recall@20': 0.789, 'Recall@50': 0.861, 'Recall@100': 0.252 },
    { id: 0, 'Recall@20': 0.407, 'Recall@50': 0.35, 'Recall@100': 0.093 },
    { id: 0, 'Recall@20': 0.422, 'Recall@50': 0.612, 'Recall@100': 0.39 },
  ],
  SVD: [
    { id: 0, 'Recall@20': 0.691, 'Recall@50': 0.762, 'Recall@100': 0.005 },
    { id: 0, 'Recall@20': 0.085, 'Recall@50': 0.528, 'Recall@100': 0.531 },
    { id: 0, 'Recall@20': 0.362, 'Recall@50': 0.221, 'Recall@100': 0.951 },
    { id: 0, 'Recall@20': 0.694, 'Recall@50': 0.147, 'Recall@100': 0.022 },
    { id: 0, 'Recall@20': 0.995, 'Recall@50': 0.086, 'Recall@100': 0.578 },
    { id: 0, 'Recall@20': 0.243, 'Recall@50': 0.787, 'Recall@100': 0.488 },
    { id: 0, 'Recall@20': 0.84, 'Recall@50': 0.053, 'Recall@100': 0.313 },
    { id: 0, 'Recall@20': 0.062, 'Recall@50': 0.264, 'Recall@100': 0.035 },
    { id: 0, 'Recall@20': 0.551, 'Recall@50': 0.99, 'Recall@100': 0.323 },
    { id: 0, 'Recall@20': 0.917, 'Recall@50': 0.894, 'Recall@100': 0.174 },
    { id: 0, 'Recall@20': 0.762, 'Recall@50': 0.524, 'Recall@100': 0.105 },
    { id: 0, 'Recall@20': 0.973, 'Recall@50': 0.713, 'Recall@100': 0.246 },
    { id: 0, 'Recall@20': 0.637, 'Recall@50': 0.318, 'Recall@100': 0.661 },
    { id: 0, 'Recall@20': 0.975, 'Recall@50': 0.151, 'Recall@100': 0.515 },
    { id: 0, 'Recall@20': 0.733, 'Recall@50': 0.972, 'Recall@100': 0.051 },
    { id: 0, 'Recall@20': 0.605, 'Recall@50': 0.273, 'Recall@100': 0.171 },
    { id: 0, 'Recall@20': 0.774, 'Recall@50': 0.575, 'Recall@100': 0.3 },
    { id: 0, 'Recall@20': 0.117, 'Recall@50': 0.579, 'Recall@100': 0.996 },
    { id: 0, 'Recall@20': 0.568, 'Recall@50': 0.102, 'Recall@100': 0.349 },
    { id: 0, 'Recall@20': 0.434, 'Recall@50': 0.575, 'Recall@100': 0.975 },
    { id: 0, 'Recall@20': 0.232, 'Recall@50': 0.936, 'Recall@100': 0.548 },
    { id: 0, 'Recall@20': 0.201, 'Recall@50': 0.65, 'Recall@100': 0.567 },
    { id: 0, 'Recall@20': 0.154, 'Recall@50': 0.825, 'Recall@100': 0.85 },
    { id: 0, 'Recall@20': 0.275, 'Recall@50': 0.924, 'Recall@100': 0.24 },
    { id: 0, 'Recall@20': 0.556, 'Recall@50': 0.365, 'Recall@100': 0.737 },
    { id: 0, 'Recall@20': 0.38, 'Recall@50': 0.4, 'Recall@100': 0.411 },
    { id: 0, 'Recall@20': 0.832, 'Recall@50': 0.459, 'Recall@100': 0.344 },
    { id: 0, 'Recall@20': 0.484, 'Recall@50': 0.754, 'Recall@100': 0.633 },
    { id: 0, 'Recall@20': 0.056, 'Recall@50': 0.291, 'Recall@100': 0.656 },
    { id: 0, 'Recall@20': 0.583, 'Recall@50': 0.12, 'Recall@100': 0.546 },
    { id: 0, 'Recall@20': 0.49, 'Recall@50': 0.426, 'Recall@100': 0.986 },
    { id: 0, 'Recall@20': 0.385, 'Recall@50': 0.989, 'Recall@100': 0.172 },
    { id: 0, 'Recall@20': 0.138, 'Recall@50': 0.692, 'Recall@100': 0.432 },
    { id: 0, 'Recall@20': 0.603, 'Recall@50': 0.795, 'Recall@100': 0.75 },
    { id: 0, 'Recall@20': 0.266, 'Recall@50': 0.79, 'Recall@100': 0.504 },
    { id: 0, 'Recall@20': 0.242, 'Recall@50': 0.051, 'Recall@100': 0.838 },
    { id: 0, 'Recall@20': 0.35, 'Recall@50': 0.575, 'Recall@100': 0.55 },
    { id: 0, 'Recall@20': 0.971, 'Recall@50': 0.696, 'Recall@100': 0.232 },
    { id: 0, 'Recall@20': 0.099, 'Recall@50': 0.123, 'Recall@100': 0.081 },
    { id: 0, 'Recall@20': 0.762, 'Recall@50': 0.674, 'Recall@100': 0.252 },
    { id: 0, 'Recall@20': 0.05, 'Recall@50': 0.181, 'Recall@100': 0.507 },
    { id: 0, 'Recall@20': 0.181, 'Recall@50': 0.045, 'Recall@100': 0.462 },
    { id: 0, 'Recall@20': 0.33, 'Recall@50': 0.972, 'Recall@100': 0.073 },
    { id: 0, 'Recall@20': 0.048, 'Recall@50': 0.484, 'Recall@100': 0.292 },
    { id: 0, 'Recall@20': 0.018, 'Recall@50': 0.239, 'Recall@100': 0.826 },
    { id: 0, 'Recall@20': 0.591, 'Recall@50': 0.309, 'Recall@100': 0.766 },
    { id: 0, 'Recall@20': 0.361, 'Recall@50': 0.806, 'Recall@100': 0.276 },
    { id: 0, 'Recall@20': 0.551, 'Recall@50': 0.512, 'Recall@100': 0.314 },
    { id: 0, 'Recall@20': 0.388, 'Recall@50': 0.612, 'Recall@100': 0.971 },
    { id: 0, 'Recall@20': 0.918, 'Recall@50': 0.963, 'Recall@100': 0.778 },
    { id: 0, 'Recall@20': 0.833, 'Recall@50': 0.901, 'Recall@100': 0.145 },
    { id: 0, 'Recall@20': 0.658, 'Recall@50': 0.654, 'Recall@100': 0.831 },
    { id: 0, 'Recall@20': 0.357, 'Recall@50': 0.56, 'Recall@100': 0.6 },
    { id: 0, 'Recall@20': 0.694, 'Recall@50': 0.002, 'Recall@100': 0.719 },
    { id: 0, 'Recall@20': 0.239, 'Recall@50': 0.922, 'Recall@100': 0.118 },
    { id: 0, 'Recall@20': 0.331, 'Recall@50': 0.667, 'Recall@100': 0.044 },
    { id: 0, 'Recall@20': 0.452, 'Recall@50': 0.697, 'Recall@100': 0.315 },
    { id: 0, 'Recall@20': 0.559, 'Recall@50': 0.87, 'Recall@100': 0.248 },
    { id: 0, 'Recall@20': 0.152, 'Recall@50': 0.607, 'Recall@100': 0.345 },
    { id: 0, 'Recall@20': 0.13, 'Recall@50': 0.324, 'Recall@100': 0.321 },
    { id: 0, 'Recall@20': 0.558, 'Recall@50': 0.916, 'Recall@100': 0.228 },
    { id: 0, 'Recall@20': 0.002, 'Recall@50': 0.428, 'Recall@100': 0.584 },
    { id: 0, 'Recall@20': 0.121, 'Recall@50': 0.625, 'Recall@100': 0.507 },
    { id: 0, 'Recall@20': 0.117, 'Recall@50': 0.505, 'Recall@100': 0.503 },
    { id: 0, 'Recall@20': 0.676, 'Recall@50': 0.31, 'Recall@100': 0.448 },
    { id: 0, 'Recall@20': 0.49, 'Recall@50': 0.283, 'Recall@100': 0.106 },
    { id: 0, 'Recall@20': 0.271, 'Recall@50': 0.9, 'Recall@100': 0.088 },
    { id: 0, 'Recall@20': 0.252, 'Recall@50': 0.274, 'Recall@100': 0.662 },
    { id: 0, 'Recall@20': 0.479, 'Recall@50': 0.453, 'Recall@100': 0.802 },
    { id: 0, 'Recall@20': 0.682, 'Recall@50': 0.383, 'Recall@100': 0.395 },
    { id: 0, 'Recall@20': 0.13, 'Recall@50': 0.931, 'Recall@100': 0.833 },
    { id: 0, 'Recall@20': 0.396, 'Recall@50': 0.351, 'Recall@100': 0.526 },
    { id: 0, 'Recall@20': 0.159, 'Recall@50': 0.861, 'Recall@100': 0.717 },
    { id: 0, 'Recall@20': 0.063, 'Recall@50': 0.018, 'Recall@100': 0.522 },
    { id: 0, 'Recall@20': 0.689, 'Recall@50': 0.791, 'Recall@100': 0.792 },
    { id: 0, 'Recall@20': 0.184, 'Recall@50': 0.314, 'Recall@100': 0.374 },
    { id: 0, 'Recall@20': 0.491, 'Recall@50': 0.709, 'Recall@100': 0.942 },
    { id: 0, 'Recall@20': 0.444, 'Recall@50': 0.299, 'Recall@100': 0.795 },
    { id: 0, 'Recall@20': 0.06, 'Recall@50': 0.449, 'Recall@100': 0.001 },
    { id: 0, 'Recall@20': 0.794, 'Recall@50': 0.967, 'Recall@100': 0.115 },
    { id: 0, 'Recall@20': 0.135, 'Recall@50': 0.967, 'Recall@100': 0.121 },
    { id: 0, 'Recall@20': 0.766, 'Recall@50': 0.089, 'Recall@100': 0.028 },
    { id: 0, 'Recall@20': 0.119, 'Recall@50': 0.097, 'Recall@100': 0.96 },
    { id: 0, 'Recall@20': 0.055, 'Recall@50': 0.203, 'Recall@100': 0.866 },
    { id: 0, 'Recall@20': 0.052, 'Recall@50': 0.814, 'Recall@100': 0.516 },
    { id: 0, 'Recall@20': 0.917, 'Recall@50': 0.829, 'Recall@100': 0.081 },
    { id: 0, 'Recall@20': 0.614, 'Recall@50': 0.609, 'Recall@100': 0.753 },
    { id: 0, 'Recall@20': 0.851, 'Recall@50': 0.733, 'Recall@100': 0.919 },
    { id: 0, 'Recall@20': 0.261, 'Recall@50': 0.445, 'Recall@100': 0.609 },
    { id: 0, 'Recall@20': 0.133, 'Recall@50': 0.862, 'Recall@100': 0.585 },
    { id: 0, 'Recall@20': 0.495, 'Recall@50': 0.676, 'Recall@100': 0.156 },
    { id: 0, 'Recall@20': 0.921, 'Recall@50': 0.813, 'Recall@100': 0.477 },
    { id: 0, 'Recall@20': 0.682, 'Recall@50': 0.664, 'Recall@100': 0.422 },
    { id: 0, 'Recall@20': 0.226, 'Recall@50': 0.3, 'Recall@100': 0.38 },
    { id: 0, 'Recall@20': 0.081, 'Recall@50': 0.971, 'Recall@100': 0.65 },
    { id: 0, 'Recall@20': 0.062, 'Recall@50': 0.066, 'Recall@100': 0.418 },
    { id: 0, 'Recall@20': 0.572, 'Recall@50': 0.387, 'Recall@100': 0.352 },
    { id: 0, 'Recall@20': 0.893, 'Recall@50': 0.183, 'Recall@100': 0.31 },
    { id: 0, 'Recall@20': 0.214, 'Recall@50': 0.706, 'Recall@100': 0.474 },
    { id: 0, 'Recall@20': 0.878, 'Recall@50': 0.458, 'Recall@100': 0.925 },
  ],
};

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function MetricsHistogramPlot({ onSelect }) {
  const [isLoading, setIsLoading] = useState(true);
  const [histData, setHistData] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [selectedMetric, setSelectedMetric] = useState('');

  const histRef = useRef();

  useEffect(() => {
    async function fetchData() {
      setIsLoading(true);
      await sleep(1000);
      setIsLoading(false);
    }
    fetchData();
    setHistData(histogramData.SVD);
    setSelectedModel('SVD');
    setSelectedMetric('Recall@20');
  }, []);

  const resetHistSelection = () => {
    Plotly.restyle(histRef.current.el, { selectedpoints: [null] });
  };

  const resetSelection = () => {
    onSelect({
      ids: [],
      points: [],
    });
  };

  const handleFilterApply = async () => {
    setIsLoading(true);
    await sleep(500);
    setIsLoading(false);
  };

  const histogramPoints = useMemo(
    () => ({
      x: histData.map((d) => d[selectedMetric]),
      meta: histData.map(({ id }) => ({ id })),
    }),
    [isLoading]
  );

  return (
    <Paper sx={{ p: 2 }}>
      <Stack direction="row" spacing={2}>
        <CategoryFilter
          label="Model"
          value={selectedModel}
          onChange={(newValue) => {
            setSelectedModel(newValue);
            // setSelectedMetric(Object.keys(histData[selectedModel][0])[0]);
            resetHistSelection();
            resetSelection();
          }}
          options={['SVD', 'KNN'].map((model) => ({
            value: model,
            label: model,
          }))}
        />
        <CategoryFilter
          label="Metric"
          value={selectedMetric}
          onBlur={handleFilterApply}
          onChange={(newValue) => {
            setSelectedMetric(newValue);
            resetHistSelection();
            resetSelection();
          }}
          options={['Recall@20', 'Recall@50'].map((metric) => ({
            value: metric,
            label: metric,
          }))}
        />
      </Stack>
      <HistogramPlot
        data={histogramPoints.x}
        meta={histogramPoints.meta}
        height={400}
        isLoading={isLoading}
        innerRef={histRef}
        onDeselect={() => {
          resetSelection();
        }}
        onSelected={(eventData) => {
          if (eventData && eventData.points.length) {
            const { points } = eventData;
            onSelect({
              ids: points.map((p) => p.customdata.id),
              points: points[0].data.selectedpoints,
            });
          }
        }}
      />
    </Paper>
  );
}

MetricsHistogramPlot.defaultProps = {};

MetricsHistogramPlot.propTypes = {
  onSelect: pt.func.isRequired,
};

export default MetricsHistogramPlot;
