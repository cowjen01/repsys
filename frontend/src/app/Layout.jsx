import React from 'react';
import Box from '@mui/material/Box';
import Toolbar from '@mui/material/Toolbar';
import MuiAppBar from '@mui/material/AppBar';
import Typography from '@mui/material/Typography';
import Switch from '@mui/material/Switch';
import FormControlLabel from '@mui/material/FormControlLabel';
import { useDispatch, useSelector } from 'react-redux';

import { toggleBuildMode, buildModeSelector } from './studioSlice';

function Layout({ children }) {
  const dispatch = useDispatch();
  const buildMode = useSelector(buildModeSelector);

  return (
    <Box sx={{ display: 'flex' }}>
      <MuiAppBar position="fixed" color="secondary" elevation={2}>
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Repsys
          </Typography>
          <FormControlLabel
            control={<Switch checked={buildMode} onChange={() => dispatch(toggleBuildMode())} />}
            label="Build mode"
          />
        </Toolbar>
      </MuiAppBar>
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          height: '100vh',
          overflow: 'auto',
          paddingTop: 4,
          paddingBottom: 4,
        }}
      >
        <Toolbar />
        {children}
      </Box>
    </Box>
  );
}

export default Layout;
