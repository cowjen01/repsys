import React from 'react';
import { Backdrop, CircularProgress } from '@mui/material';

function PlotLoader() {
  return (
    <Backdrop
      sx={{
        position: 'absolute',
        color: '#000',
        backgroundColor: 'rgba(255,255,255,0.6)',
        zIndex: 100,
      }}
      open
    >
      <CircularProgress color="inherit" />
    </Backdrop>
  );
}

export default PlotLoader;
