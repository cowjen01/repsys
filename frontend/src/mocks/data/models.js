export const models = {
  knn: {
    params: {
      neighbors: {
        field: 'number',
        default: 5,
      },
      category: {
        field: 'select',
        options: ['Adventure', 'Animation', 'Children', 'Comedy', 'Musical', 'Crime', 'Thriller'],
      },
      normalize: {
        field: 'checkbox',
        default: false,
      },
    },
  },
  svd: {
    params: {
      factor: {
        field: 'number',
      },
      foo: {
        field: 'checkbox',
      },
    },
  },
};

export const summaryMetrics = {
  results: {
    knn: {
      current: {
        'Recall@20': 0.3,
        'Recall@50': 0.4,
        'Recall@100': 0.32,
        'NDCG@100': 0.1,
        'Coverage@20': 0.3,
        'Coverage@50': 0.8,
        'Coverage@100': 0.2,
        'Novelty@10': 0.65,
        MAE: 0.01,
        RMSE: 0.003,
        'Diversity@20': 0.56,
        'Diversity@50': 0.7,
      },
      previous: {
        'Recall@20': 0.2,
        'Recall@50': 0.5,
        'NDCG@100': 0.3,
        'Coverage@20': 0.1,
        'Coverage@50': 0.7,
        'Coverage@100': 0.15,
      },
    },
    svd: {
      current: {
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
  },
};

export const itemMetrics = {
  knn: [
    { id: 1, popularity: 0.0039, tfidfpop: 0.0108 },
    { id: 2, popularity: 0.2069, tfidfpop: 0.4858 },
    { id: 3, popularity: 0.9807, tfidfpop: 0.1974 },
    { id: 4, popularity: 0.9327, tfidfpop: 0.4394 },
    { id: 5, popularity: 0.5728, tfidfpop: 0.5202 },
    { id: 6, popularity: 0.9348, tfidfpop: 0.4627 },
    { id: 7, popularity: 0.0833, tfidfpop: 0.6568 },
    { id: 8, popularity: 0.0021, tfidfpop: 0.1639 },
    { id: 9, popularity: 0.1515, tfidfpop: 0.3813 },
    { id: 10, popularity: 0.2351, tfidfpop: 0.1327 },
    { id: 11, popularity: 0.2182, tfidfpop: 0.5095 },
    { id: 12, popularity: 0.8556, tfidfpop: 0.2999 },
    { id: 13, popularity: 0.509, tfidfpop: 0.9634 },
    { id: 14, popularity: 0.4224, tfidfpop: 0.6991 },
    { id: 15, popularity: 0.301, tfidfpop: 0.7376 },
    { id: 16, popularity: 0.2544, tfidfpop: 0.9247 },
    { id: 17, popularity: 0.5562, tfidfpop: 0.331 },
    { id: 18, popularity: 0.5018, tfidfpop: 0.8685 },
    { id: 19, popularity: 0.1459, tfidfpop: 0.409 },
    { id: 20, popularity: 0.1463, tfidfpop: 0.7284 },
    { id: 21, popularity: 0.1358, tfidfpop: 0.9669 },
    { id: 22, popularity: 0.7344, tfidfpop: 0.7076 },
    { id: 23, popularity: 0.5239, tfidfpop: 0.0849 },
    { id: 24, popularity: 0.7277, tfidfpop: 0.8635 },
    { id: 25, popularity: 0.9081, tfidfpop: 0.2732 },
    { id: 26, popularity: 0.9081, tfidfpop: 0.9601 },
    { id: 27, popularity: 0.6471, tfidfpop: 0.0283 },
    { id: 28, popularity: 0.7668, tfidfpop: 0.1348 },
    { id: 29, popularity: 0.6157, tfidfpop: 0.67 },
    { id: 30, popularity: 0.2613, tfidfpop: 0.2734 },
    { id: 31, popularity: 0.2892, tfidfpop: 0.7427 },
    { id: 32, popularity: 0.9154, tfidfpop: 0.9575 },
    { id: 33, popularity: 0.049, tfidfpop: 0.9151 },
    { id: 34, popularity: 0.9689, tfidfpop: 0.2116 },
    { id: 35, popularity: 0.0494, tfidfpop: 0.2956 },
    { id: 36, popularity: 0.7345, tfidfpop: 0.1674 },
    { id: 37, popularity: 0.9322, tfidfpop: 0.7054 },
    { id: 38, popularity: 0.9374, tfidfpop: 0.0987 },
    { id: 39, popularity: 0.7594, tfidfpop: 0.8492 },
    { id: 40, popularity: 0.2411, tfidfpop: 0.0247 },
    { id: 41, popularity: 0.4945, tfidfpop: 0.3498 },
    { id: 42, popularity: 0.2743, tfidfpop: 0.8307 },
    { id: 43, popularity: 0.3536, tfidfpop: 0.555 },
    { id: 44, popularity: 0.2254, tfidfpop: 0.53 },
    { id: 45, popularity: 0.6917, tfidfpop: 0.0323 },
    { id: 46, popularity: 0.8931, tfidfpop: 0.851 },
    { id: 47, popularity: 0.031, tfidfpop: 0.0634 },
    { id: 48, popularity: 0.1201, tfidfpop: 0.7607 },
    { id: 49, popularity: 0.7796, tfidfpop: 0.0704 },
    { id: 50, popularity: 0.8463, tfidfpop: 0.7518 },
    { id: 51, popularity: 0.4382, tfidfpop: 0.5857 },
    { id: 52, popularity: 0.9061, tfidfpop: 0.8198 },
    { id: 53, popularity: 0.5178, tfidfpop: 0.4343 },
    { id: 54, popularity: 0.9651, tfidfpop: 0.5897 },
    { id: 55, popularity: 0.5433, tfidfpop: 0.3342 },
    { id: 56, popularity: 0.0146, tfidfpop: 0.4186 },
    { id: 57, popularity: 0.1895, tfidfpop: 0.7143 },
    { id: 58, popularity: 0.3405, tfidfpop: 0.7277 },
    { id: 59, popularity: 0.1833, tfidfpop: 0.5072 },
    { id: 60, popularity: 0.7926, tfidfpop: 0.0995 },
    { id: 61, popularity: 0.1908, tfidfpop: 0.2992 },
    { id: 62, popularity: 0.4553, tfidfpop: 0.7423 },
    { id: 63, popularity: 0.4865, tfidfpop: 0.5954 },
    { id: 64, popularity: 0.8856, tfidfpop: 0.408 },
    { id: 65, popularity: 0.3158, tfidfpop: 0.4006 },
    { id: 66, popularity: 0.9687, tfidfpop: 0.21 },
    { id: 67, popularity: 0.2513, tfidfpop: 0.973 },
    { id: 68, popularity: 0.3815, tfidfpop: 0.6326 },
    { id: 69, popularity: 0.6643, tfidfpop: 0.7705 },
    { id: 70, popularity: 0.3859, tfidfpop: 0.0914 },
    { id: 71, popularity: 0.7279, tfidfpop: 0.9741 },
    { id: 72, popularity: 0.1218, tfidfpop: 0.4981 },
    { id: 73, popularity: 0.4233, tfidfpop: 0.7558 },
    { id: 74, popularity: 0.3578, tfidfpop: 0.1984 },
    { id: 75, popularity: 0.9531, tfidfpop: 0.3926 },
    { id: 76, popularity: 0.6587, tfidfpop: 0.6204 },
    { id: 77, popularity: 0.7788, tfidfpop: 0.2942 },
    { id: 78, popularity: 0.2018, tfidfpop: 0.0098 },
    { id: 79, popularity: 0.0628, tfidfpop: 0.532 },
    { id: 80, popularity: 0.7653, tfidfpop: 0.0492 },
    { id: 81, popularity: 0.8052, tfidfpop: 0.2584 },
    { id: 82, popularity: 0.0317, tfidfpop: 0.5014 },
    { id: 83, popularity: 0.8936, tfidfpop: 0.2992 },
    { id: 84, popularity: 0.8123, tfidfpop: 0.6678 },
    { id: 85, popularity: 0.4311, tfidfpop: 0.7969 },
    { id: 86, popularity: 0.5449, tfidfpop: 0.2665 },
    { id: 87, popularity: 0.226, tfidfpop: 0.8598 },
    { id: 88, popularity: 0.2341, tfidfpop: 0.956 },
    { id: 89, popularity: 0.9051, tfidfpop: 0.1997 },
    { id: 90, popularity: 0.7317, tfidfpop: 0.6128 },
    { id: 91, popularity: 0.7897, tfidfpop: 0.7685 },
    { id: 92, popularity: 0.4699, tfidfpop: 0.3346 },
    { id: 93, popularity: 0.3793, tfidfpop: 0.6316 },
    { id: 94, popularity: 0.1635, tfidfpop: 0.3039 },
    { id: 95, popularity: 0.2976, tfidfpop: 0.062 },
    { id: 96, popularity: 0.1199, tfidfpop: 0.4166 },
    { id: 97, popularity: 0.5132, tfidfpop: 0.3456 },
    { id: 98, popularity: 0.9016, tfidfpop: 0.5581 },
    { id: 99, popularity: 0.0688, tfidfpop: 0.8405 },
    { id: 100, popularity: 0.4418, tfidfpop: 0.3825 },
  ],
};

export const userMetrics = {
  knn: [
    { id: 101, 'Recall@20': 0.298, 'Recall@50': 0.895, 'Recall@100': 0.877 },
    { id: 102, 'Recall@20': 0.278, 'Recall@50': 0.1, 'Recall@100': 0.0 },
    { id: 103, 'Recall@20': 0.617, 'Recall@50': 0.382, 'Recall@100': 0.886 },
    { id: 104, 'Recall@20': 0.204, 'Recall@50': 0.755, 'Recall@100': 0.888 },
    { id: 105, 'Recall@20': 0.328, 'Recall@50': 0.854, 'Recall@100': 0.801 },
    { id: 106, 'Recall@20': 0.474, 'Recall@50': 0.893, 'Recall@100': 0.78 },
    { id: 107, 'Recall@20': 0.602, 'Recall@50': 0.668, 'Recall@100': 0.908 },
    { id: 108, 'Recall@20': 0.238, 'Recall@50': 0.276, 'Recall@100': 0.211 },
    { id: 109, 'Recall@20': 0.22, 'Recall@50': 0.951, 'Recall@100': 0.589 },
    { id: 110, 'Recall@20': 0.273, 'Recall@50': 0.269, 'Recall@100': 0.009 },
    { id: 111, 'Recall@20': 0.718, 'Recall@50': 0.265, 'Recall@100': 0.572 },
    { id: 112, 'Recall@20': 0.653, 'Recall@50': 0.383, 'Recall@100': 0.923 },
    { id: 113, 'Recall@20': 0.469, 'Recall@50': 0.791, 'Recall@100': 0.822 },
    { id: 114, 'Recall@20': 0.307, 'Recall@50': 0.197, 'Recall@100': 0.743 },
    { id: 115, 'Recall@20': 0.608, 'Recall@50': 0.624, 'Recall@100': 0.806 },
    { id: 116, 'Recall@20': 0.668, 'Recall@50': 0.529, 'Recall@100': 0.045 },
    { id: 117, 'Recall@20': 0.631, 'Recall@50': 0.586, 'Recall@100': 0.287 },
    { id: 118, 'Recall@20': 0.113, 'Recall@50': 0.999, 'Recall@100': 0.956 },
    { id: 119, 'Recall@20': 0.911, 'Recall@50': 0.125, 'Recall@100': 0.285 },
    { id: 120, 'Recall@20': 0.902, 'Recall@50': 0.13, 'Recall@100': 0.617 },
    { id: 121, 'Recall@20': 0.027, 'Recall@50': 0.403, 'Recall@100': 0.397 },
    { id: 122, 'Recall@20': 0.876, 'Recall@50': 0.161, 'Recall@100': 0.722 },
    { id: 123, 'Recall@20': 0.034, 'Recall@50': 0.428, 'Recall@100': 0.216 },
    { id: 124, 'Recall@20': 0.74, 'Recall@50': 0.797, 'Recall@100': 0.381 },
    { id: 125, 'Recall@20': 0.507, 'Recall@50': 0.635, 'Recall@100': 0.734 },
    { id: 126, 'Recall@20': 0.719, 'Recall@50': 0.826, 'Recall@100': 0.28 },
    { id: 127, 'Recall@20': 0.315, 'Recall@50': 0.816, 'Recall@100': 0.408 },
    { id: 128, 'Recall@20': 0.813, 'Recall@50': 0.151, 'Recall@100': 0.852 },
    { id: 129, 'Recall@20': 0.618, 'Recall@50': 0.57, 'Recall@100': 0.51 },
    { id: 130, 'Recall@20': 0.341, 'Recall@50': 0.15, 'Recall@100': 0.001 },
    { id: 131, 'Recall@20': 0.412, 'Recall@50': 0.69, 'Recall@100': 0.931 },
    { id: 132, 'Recall@20': 0.719, 'Recall@50': 0.382, 'Recall@100': 0.613 },
    { id: 133, 'Recall@20': 0.566, 'Recall@50': 0.379, 'Recall@100': 0.817 },
    { id: 134, 'Recall@20': 0.727, 'Recall@50': 0.832, 'Recall@100': 0.603 },
    { id: 135, 'Recall@20': 0.048, 'Recall@50': 0.099, 'Recall@100': 0.939 },
    { id: 136, 'Recall@20': 0.352, 'Recall@50': 0.531, 'Recall@100': 0.921 },
    { id: 137, 'Recall@20': 0.259, 'Recall@50': 0.187, 'Recall@100': 0.233 },
    { id: 138, 'Recall@20': 0.781, 'Recall@50': 0.238, 'Recall@100': 0.36 },
    { id: 139, 'Recall@20': 0.931, 'Recall@50': 0.533, 'Recall@100': 0.786 },
    { id: 140, 'Recall@20': 0.864, 'Recall@50': 0.828, 'Recall@100': 0.438 },
    { id: 141, 'Recall@20': 0.432, 'Recall@50': 0.296, 'Recall@100': 0.679 },
    { id: 142, 'Recall@20': 0.497, 'Recall@50': 0.349, 'Recall@100': 0.35 },
    { id: 143, 'Recall@20': 0.614, 'Recall@50': 0.577, 'Recall@100': 0.474 },
    { id: 144, 'Recall@20': 0.095, 'Recall@50': 0.051, 'Recall@100': 0.089 },
    { id: 145, 'Recall@20': 0.542, 'Recall@50': 0.884, 'Recall@100': 0.116 },
    { id: 146, 'Recall@20': 0.303, 'Recall@50': 0.263, 'Recall@100': 0.211 },
    { id: 147, 'Recall@20': 0.153, 'Recall@50': 0.95, 'Recall@100': 0.561 },
    { id: 148, 'Recall@20': 0.371, 'Recall@50': 0.628, 'Recall@100': 0.516 },
    { id: 149, 'Recall@20': 0.808, 'Recall@50': 0.661, 'Recall@100': 0.374 },
    { id: 150, 'Recall@20': 0.7, 'Recall@50': 0.866, 'Recall@100': 0.383 },
    { id: 151, 'Recall@20': 0.255, 'Recall@50': 0.426, 'Recall@100': 0.586 },
    { id: 152, 'Recall@20': 0.149, 'Recall@50': 0.905, 'Recall@100': 0.145 },
    { id: 153, 'Recall@20': 0.127, 'Recall@50': 0.574, 'Recall@100': 0.16 },
    { id: 154, 'Recall@20': 0.761, 'Recall@50': 0.3, 'Recall@100': 0.08 },
    { id: 155, 'Recall@20': 0.506, 'Recall@50': 0.089, 'Recall@100': 0.817 },
    { id: 156, 'Recall@20': 0.506, 'Recall@50': 0.137, 'Recall@100': 0.319 },
    { id: 157, 'Recall@20': 0.241, 'Recall@50': 0.347, 'Recall@100': 0.765 },
    { id: 158, 'Recall@20': 0.364, 'Recall@50': 0.609, 'Recall@100': 0.161 },
    { id: 159, 'Recall@20': 0.46, 'Recall@50': 0.389, 'Recall@100': 0.35 },
    { id: 160, 'Recall@20': 0.224, 'Recall@50': 0.71, 'Recall@100': 0.448 },
    { id: 161, 'Recall@20': 0.088, 'Recall@50': 0.791, 'Recall@100': 0.07 },
    { id: 162, 'Recall@20': 0.323, 'Recall@50': 0.655, 'Recall@100': 0.586 },
    { id: 163, 'Recall@20': 0.253, 'Recall@50': 0.922, 'Recall@100': 0.135 },
    { id: 164, 'Recall@20': 0.112, 'Recall@50': 0.36, 'Recall@100': 0.753 },
    { id: 165, 'Recall@20': 0.925, 'Recall@50': 0.908, 'Recall@100': 0.769 },
    { id: 166, 'Recall@20': 0.544, 'Recall@50': 0.097, 'Recall@100': 0.528 },
    { id: 167, 'Recall@20': 0.821, 'Recall@50': 0.856, 'Recall@100': 0.241 },
    { id: 168, 'Recall@20': 0.156, 'Recall@50': 0.279, 'Recall@100': 0.322 },
    { id: 169, 'Recall@20': 0.687, 'Recall@50': 0.638, 'Recall@100': 0.475 },
    { id: 170, 'Recall@20': 0.611, 'Recall@50': 0.592, 'Recall@100': 0.643 },
    { id: 171, 'Recall@20': 0.787, 'Recall@50': 0.039, 'Recall@100': 0.061 },
    { id: 172, 'Recall@20': 0.247, 'Recall@50': 0.777, 'Recall@100': 0.154 },
    { id: 173, 'Recall@20': 0.151, 'Recall@50': 0.148, 'Recall@100': 0.182 },
    { id: 174, 'Recall@20': 0.651, 'Recall@50': 0.288, 'Recall@100': 0.71 },
    { id: 175, 'Recall@20': 0.388, 'Recall@50': 0.033, 'Recall@100': 0.074 },
    { id: 176, 'Recall@20': 0.716, 'Recall@50': 0.759, 'Recall@100': 0.871 },
    { id: 177, 'Recall@20': 0.624, 'Recall@50': 0.173, 'Recall@100': 0.224 },
    { id: 178, 'Recall@20': 0.148, 'Recall@50': 0.952, 'Recall@100': 0.348 },
    { id: 179, 'Recall@20': 0.037, 'Recall@50': 0.039, 'Recall@100': 0.403 },
    { id: 180, 'Recall@20': 0.175, 'Recall@50': 0.39, 'Recall@100': 0.642 },
    { id: 181, 'Recall@20': 0.008, 'Recall@50': 0.026, 'Recall@100': 0.283 },
    { id: 182, 'Recall@20': 0.9, 'Recall@50': 0.32, 'Recall@100': 0.437 },
    { id: 183, 'Recall@20': 0.715, 'Recall@50': 0.92, 'Recall@100': 0.992 },
    { id: 184, 'Recall@20': 0.91, 'Recall@50': 0.411, 'Recall@100': 0.989 },
    { id: 185, 'Recall@20': 0.311, 'Recall@50': 0.082, 'Recall@100': 0.756 },
    { id: 186, 'Recall@20': 0.782, 'Recall@50': 0.172, 'Recall@100': 0.845 },
    { id: 187, 'Recall@20': 0.596, 'Recall@50': 0.828, 'Recall@100': 0.45 },
    { id: 188, 'Recall@20': 0.584, 'Recall@50': 0.099, 'Recall@100': 0.979 },
    { id: 189, 'Recall@20': 0.975, 'Recall@50': 0.176, 'Recall@100': 0.133 },
    { id: 190, 'Recall@20': 0.467, 'Recall@50': 0.795, 'Recall@100': 0.469 },
    { id: 191, 'Recall@20': 0.947, 'Recall@50': 0.214, 'Recall@100': 0.362 },
    { id: 192, 'Recall@20': 0.681, 'Recall@50': 0.666, 'Recall@100': 0.295 },
    { id: 193, 'Recall@20': 0.913, 'Recall@50': 0.157, 'Recall@100': 0.65 },
    { id: 194, 'Recall@20': 0.656, 'Recall@50': 0.011, 'Recall@100': 0.221 },
    { id: 195, 'Recall@20': 0.115, 'Recall@50': 0.968, 'Recall@100': 0.025 },
    { id: 196, 'Recall@20': 0.716, 'Recall@50': 0.439, 'Recall@100': 0.22 },
    { id: 197, 'Recall@20': 0.513, 'Recall@50': 0.468, 'Recall@100': 0.692 },
    { id: 198, 'Recall@20': 0.789, 'Recall@50': 0.861, 'Recall@100': 0.252 },
    { id: 199, 'Recall@20': 0.407, 'Recall@50': 0.35, 'Recall@100': 0.093 },
    { id: 200, 'Recall@20': 0.422, 'Recall@50': 0.612, 'Recall@100': 0.39 },
  ],
  svd: [
    { id: 101, 'Recall@20': 0.691, 'Recall@50': 0.762, 'Recall@100': 0.005 },
    { id: 102, 'Recall@20': 0.085, 'Recall@50': 0.528, 'Recall@100': 0.531 },
    { id: 103, 'Recall@20': 0.362, 'Recall@50': 0.221, 'Recall@100': 0.951 },
    { id: 104, 'Recall@20': 0.694, 'Recall@50': 0.147, 'Recall@100': 0.022 },
    { id: 105, 'Recall@20': 0.995, 'Recall@50': 0.086, 'Recall@100': 0.578 },
    { id: 106, 'Recall@20': 0.243, 'Recall@50': 0.787, 'Recall@100': 0.488 },
    { id: 107, 'Recall@20': 0.84, 'Recall@50': 0.053, 'Recall@100': 0.313 },
    { id: 108, 'Recall@20': 0.062, 'Recall@50': 0.264, 'Recall@100': 0.035 },
    { id: 109, 'Recall@20': 0.551, 'Recall@50': 0.99, 'Recall@100': 0.323 },
    { id: 110, 'Recall@20': 0.917, 'Recall@50': 0.894, 'Recall@100': 0.174 },
    { id: 111, 'Recall@20': 0.762, 'Recall@50': 0.524, 'Recall@100': 0.105 },
    { id: 112, 'Recall@20': 0.973, 'Recall@50': 0.713, 'Recall@100': 0.246 },
    { id: 113, 'Recall@20': 0.637, 'Recall@50': 0.318, 'Recall@100': 0.661 },
    { id: 114, 'Recall@20': 0.975, 'Recall@50': 0.151, 'Recall@100': 0.515 },
    { id: 115, 'Recall@20': 0.733, 'Recall@50': 0.972, 'Recall@100': 0.051 },
    { id: 116, 'Recall@20': 0.605, 'Recall@50': 0.273, 'Recall@100': 0.171 },
    { id: 117, 'Recall@20': 0.774, 'Recall@50': 0.575, 'Recall@100': 0.3 },
    { id: 118, 'Recall@20': 0.117, 'Recall@50': 0.579, 'Recall@100': 0.996 },
    { id: 119, 'Recall@20': 0.568, 'Recall@50': 0.102, 'Recall@100': 0.349 },
    { id: 120, 'Recall@20': 0.434, 'Recall@50': 0.575, 'Recall@100': 0.975 },
    { id: 121, 'Recall@20': 0.232, 'Recall@50': 0.936, 'Recall@100': 0.548 },
    { id: 122, 'Recall@20': 0.201, 'Recall@50': 0.65, 'Recall@100': 0.567 },
    { id: 123, 'Recall@20': 0.154, 'Recall@50': 0.825, 'Recall@100': 0.85 },
    { id: 124, 'Recall@20': 0.275, 'Recall@50': 0.924, 'Recall@100': 0.24 },
    { id: 125, 'Recall@20': 0.556, 'Recall@50': 0.365, 'Recall@100': 0.737 },
    { id: 126, 'Recall@20': 0.38, 'Recall@50': 0.4, 'Recall@100': 0.411 },
    { id: 127, 'Recall@20': 0.832, 'Recall@50': 0.459, 'Recall@100': 0.344 },
    { id: 128, 'Recall@20': 0.484, 'Recall@50': 0.754, 'Recall@100': 0.633 },
    { id: 129, 'Recall@20': 0.056, 'Recall@50': 0.291, 'Recall@100': 0.656 },
    { id: 130, 'Recall@20': 0.583, 'Recall@50': 0.12, 'Recall@100': 0.546 },
    { id: 131, 'Recall@20': 0.49, 'Recall@50': 0.426, 'Recall@100': 0.986 },
    { id: 132, 'Recall@20': 0.385, 'Recall@50': 0.989, 'Recall@100': 0.172 },
    { id: 133, 'Recall@20': 0.138, 'Recall@50': 0.692, 'Recall@100': 0.432 },
    { id: 134, 'Recall@20': 0.603, 'Recall@50': 0.795, 'Recall@100': 0.75 },
    { id: 135, 'Recall@20': 0.266, 'Recall@50': 0.79, 'Recall@100': 0.504 },
    { id: 136, 'Recall@20': 0.242, 'Recall@50': 0.051, 'Recall@100': 0.838 },
    { id: 137, 'Recall@20': 0.35, 'Recall@50': 0.575, 'Recall@100': 0.55 },
    { id: 138, 'Recall@20': 0.971, 'Recall@50': 0.696, 'Recall@100': 0.232 },
    { id: 139, 'Recall@20': 0.099, 'Recall@50': 0.123, 'Recall@100': 0.081 },
    { id: 140, 'Recall@20': 0.762, 'Recall@50': 0.674, 'Recall@100': 0.252 },
    { id: 141, 'Recall@20': 0.05, 'Recall@50': 0.181, 'Recall@100': 0.507 },
    { id: 142, 'Recall@20': 0.181, 'Recall@50': 0.045, 'Recall@100': 0.462 },
    { id: 143, 'Recall@20': 0.33, 'Recall@50': 0.972, 'Recall@100': 0.073 },
    { id: 144, 'Recall@20': 0.048, 'Recall@50': 0.484, 'Recall@100': 0.292 },
    { id: 145, 'Recall@20': 0.018, 'Recall@50': 0.239, 'Recall@100': 0.826 },
    { id: 146, 'Recall@20': 0.591, 'Recall@50': 0.309, 'Recall@100': 0.766 },
    { id: 147, 'Recall@20': 0.361, 'Recall@50': 0.806, 'Recall@100': 0.276 },
    { id: 148, 'Recall@20': 0.551, 'Recall@50': 0.512, 'Recall@100': 0.314 },
    { id: 149, 'Recall@20': 0.388, 'Recall@50': 0.612, 'Recall@100': 0.971 },
    { id: 150, 'Recall@20': 0.918, 'Recall@50': 0.963, 'Recall@100': 0.778 },
    { id: 151, 'Recall@20': 0.833, 'Recall@50': 0.901, 'Recall@100': 0.145 },
    { id: 152, 'Recall@20': 0.658, 'Recall@50': 0.654, 'Recall@100': 0.831 },
    { id: 153, 'Recall@20': 0.357, 'Recall@50': 0.56, 'Recall@100': 0.6 },
    { id: 154, 'Recall@20': 0.694, 'Recall@50': 0.002, 'Recall@100': 0.719 },
    { id: 155, 'Recall@20': 0.239, 'Recall@50': 0.922, 'Recall@100': 0.118 },
    { id: 156, 'Recall@20': 0.331, 'Recall@50': 0.667, 'Recall@100': 0.044 },
    { id: 157, 'Recall@20': 0.452, 'Recall@50': 0.697, 'Recall@100': 0.315 },
    { id: 158, 'Recall@20': 0.559, 'Recall@50': 0.87, 'Recall@100': 0.248 },
    { id: 159, 'Recall@20': 0.152, 'Recall@50': 0.607, 'Recall@100': 0.345 },
    { id: 160, 'Recall@20': 0.13, 'Recall@50': 0.324, 'Recall@100': 0.321 },
    { id: 161, 'Recall@20': 0.558, 'Recall@50': 0.916, 'Recall@100': 0.228 },
    { id: 162, 'Recall@20': 0.002, 'Recall@50': 0.428, 'Recall@100': 0.584 },
    { id: 163, 'Recall@20': 0.121, 'Recall@50': 0.625, 'Recall@100': 0.507 },
    { id: 164, 'Recall@20': 0.117, 'Recall@50': 0.505, 'Recall@100': 0.503 },
    { id: 165, 'Recall@20': 0.676, 'Recall@50': 0.31, 'Recall@100': 0.448 },
    { id: 166, 'Recall@20': 0.49, 'Recall@50': 0.283, 'Recall@100': 0.106 },
    { id: 167, 'Recall@20': 0.271, 'Recall@50': 0.9, 'Recall@100': 0.088 },
    { id: 168, 'Recall@20': 0.252, 'Recall@50': 0.274, 'Recall@100': 0.662 },
    { id: 169, 'Recall@20': 0.479, 'Recall@50': 0.453, 'Recall@100': 0.802 },
    { id: 170, 'Recall@20': 0.682, 'Recall@50': 0.383, 'Recall@100': 0.395 },
    { id: 171, 'Recall@20': 0.13, 'Recall@50': 0.931, 'Recall@100': 0.833 },
    { id: 172, 'Recall@20': 0.396, 'Recall@50': 0.351, 'Recall@100': 0.526 },
    { id: 173, 'Recall@20': 0.159, 'Recall@50': 0.861, 'Recall@100': 0.717 },
    { id: 174, 'Recall@20': 0.063, 'Recall@50': 0.018, 'Recall@100': 0.522 },
    { id: 175, 'Recall@20': 0.689, 'Recall@50': 0.791, 'Recall@100': 0.792 },
    { id: 176, 'Recall@20': 0.184, 'Recall@50': 0.314, 'Recall@100': 0.374 },
    { id: 177, 'Recall@20': 0.491, 'Recall@50': 0.709, 'Recall@100': 0.942 },
    { id: 178, 'Recall@20': 0.444, 'Recall@50': 0.299, 'Recall@100': 0.795 },
    { id: 179, 'Recall@20': 0.06, 'Recall@50': 0.449, 'Recall@100': 0.001 },
    { id: 180, 'Recall@20': 0.794, 'Recall@50': 0.967, 'Recall@100': 0.115 },
    { id: 181, 'Recall@20': 0.135, 'Recall@50': 0.967, 'Recall@100': 0.121 },
    { id: 182, 'Recall@20': 0.766, 'Recall@50': 0.089, 'Recall@100': 0.028 },
    { id: 183, 'Recall@20': 0.119, 'Recall@50': 0.097, 'Recall@100': 0.96 },
    { id: 184, 'Recall@20': 0.055, 'Recall@50': 0.203, 'Recall@100': 0.866 },
    { id: 185, 'Recall@20': 0.052, 'Recall@50': 0.814, 'Recall@100': 0.516 },
    { id: 186, 'Recall@20': 0.917, 'Recall@50': 0.829, 'Recall@100': 0.081 },
    { id: 187, 'Recall@20': 0.614, 'Recall@50': 0.609, 'Recall@100': 0.753 },
    { id: 188, 'Recall@20': 0.851, 'Recall@50': 0.733, 'Recall@100': 0.919 },
    { id: 189, 'Recall@20': 0.261, 'Recall@50': 0.445, 'Recall@100': 0.609 },
    { id: 190, 'Recall@20': 0.133, 'Recall@50': 0.862, 'Recall@100': 0.585 },
    { id: 191, 'Recall@20': 0.495, 'Recall@50': 0.676, 'Recall@100': 0.156 },
    { id: 192, 'Recall@20': 0.921, 'Recall@50': 0.813, 'Recall@100': 0.477 },
    { id: 193, 'Recall@20': 0.682, 'Recall@50': 0.664, 'Recall@100': 0.422 },
    { id: 194, 'Recall@20': 0.226, 'Recall@50': 0.3, 'Recall@100': 0.38 },
    { id: 195, 'Recall@20': 0.081, 'Recall@50': 0.971, 'Recall@100': 0.65 },
    { id: 196, 'Recall@20': 0.062, 'Recall@50': 0.066, 'Recall@100': 0.418 },
    { id: 197, 'Recall@20': 0.572, 'Recall@50': 0.387, 'Recall@100': 0.352 },
    { id: 198, 'Recall@20': 0.893, 'Recall@50': 0.183, 'Recall@100': 0.31 },
    { id: 199, 'Recall@20': 0.214, 'Recall@50': 0.706, 'Recall@100': 0.474 },
    { id: 200, 'Recall@20': 0.878, 'Recall@50': 0.458, 'Recall@100': 0.925 },
  ],
};
