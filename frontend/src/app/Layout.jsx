import React, { useMemo } from 'react';
import pt from 'prop-types';
import Box from '@mui/material/Box';
import Toolbar from '@mui/material/Toolbar';
import MuiAppBar from '@mui/material/AppBar';
import Typography from '@mui/material/Typography';
import Switch from '@mui/material/Switch';
import FormControlLabel from '@mui/material/FormControlLabel';
import { useDispatch, useSelector } from 'react-redux';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import * as colors from '@mui/material/colors';
import BubbleChartIcon from '@mui/icons-material/BubbleChart';

import {
  toggleBuildMode,
  buildModeSelector,
  darkModeSelector,
  toggleDarkMode,
} from './studioSlice';

function Layout({ children }) {
  const dispatch = useDispatch();
  const buildMode = useSelector(buildModeSelector);
  const darkMode = useSelector(darkModeSelector);

  const theme = useMemo(
    () =>
      createTheme({
        palette: {
          ...(!darkMode
            ? {
                primary: colors.amber,
                secondary: {
                  main: colors.grey[700],
                },
              }
            : {
                mode: 'dark',
              }),
        },
      }),
    [darkMode]
  );

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ display: 'flex' }}>
        <MuiAppBar position="fixed" elevation={2}>
          <Toolbar>
            <BubbleChartIcon sx={{ marginRight: 1 }} />
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              REPSYS
            </Typography>
            <FormControlLabel
              control={
                <Switch
                  color="secondary"
                  checked={buildMode}
                  onChange={() => dispatch(toggleBuildMode())}
                />
              }
              label="Build mode"
            />
            <FormControlLabel
              sx={{
                marginLeft: 1,
              }}
              control={<Switch checked={darkMode} onChange={() => dispatch(toggleDarkMode())} />}
              label="Dark mode"
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
    </ThemeProvider>
  );
}

Layout.propTypes = {
  children: pt.node.isRequired,
};

export default Layout;
