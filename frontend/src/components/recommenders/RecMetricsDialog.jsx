// import React, { useMemo } from 'react';
// // import pt from 'prop-types';
// import Box from '@mui/material/Box';
// import Grid from '@mui/material/Grid';

// const data = [
//   {
//     name: 'Recall@20',
//     uv: 0.419,
//     pv: 0.448,
//   },
//   {
//     name: 'Recall@50',
//     uv: 0.387,
//     pv: 0.414,
//   },
//   {
//     name: 'NDCG@100',
//     uv: 0.524,
//     pv: 0.552,
//   },
// ];

// const data2 = [
//   {
//     uv: 0.419,
//   },
//   {
//     uv: 0.387,
//   },
//   {
//     uv: 0.524,
//   },
//   {
//     uv: 0.287,
//   },
//   {
//     uv: 0.124,
//   },
// ];

function RecMetricsDialog() {
  // const benchmarkData = useMemo(() => data.map((x) => [x.name, x.uv, x.pv]), [data]);
  // const popularityData = useMemo(() => data2.map((x) => [null, x.uv]), [data2]);

  return null;

  // return (
  //   <Box
  //     sx={{
  //       p: 2,
  //     }}
  //   >
  //     <Grid container spacing={2} justifyContent="center" flexDirection="row">
  //       <Grid item>
  //         <Chart
  //           width={500}
  //           height={300}
  //           chartType="Bar"
  //           loader={<div>Loading Chart</div>}
  //           data={[['Method', 'Baseline', 'KNN'], ...benchmarkData]}
  //           // options={{
  //           //   chart: {
  //           //     title: 'Model Benchmarks',
  //           //     subtitle: 'Comparation of the current model against baseline',
  //           //   },
  //           // }}
  //         />
  //       </Grid>
  //       <Grid item>
  //         <Chart
  //           width={500}
  //           height={300}
  //           chartType="Histogram"
  //           loader={<div>Loading Chart</div>}
  //           data={[['foo', 'goo'], ...popularityData]}
  //           options={{
  //             legend: { position: 'none' },
  //             histogram: {
  //               hideBucketItems: true,
  //             },
  //           }}
  //         />
  //       </Grid>
  //     </Grid>
  //   </Box>
  // );
}

export default RecMetricsDialog;
