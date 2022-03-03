import React from 'react';
import { Backdrop, CircularProgress } from '@mui/material';
import { useTheme } from '@mui/material/styles';

function PlotLoader() {
  const theme = useTheme();

  return (
    <Backdrop
      sx={{
        position: 'absolute',
        color: theme.palette.text.primary,
        backgroundColor:
          theme.palette.mode === 'dark' ? 'rgba(0,0,0,0.6)' : 'rgba(255,255,255,0.6)',
        zIndex: 100,
      }}
      open
    >
      <CircularProgress color="inherit" />
    </Backdrop>
  );
}

export default PlotLoader;
