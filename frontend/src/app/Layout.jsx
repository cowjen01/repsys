import React, { useMemo, useState } from 'react';
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
import Button from '@mui/material/Button';
import MenuItem from '@mui/material/MenuItem';
import Menu from '@mui/material/Menu';
import Grid from '@mui/material/Grid';
import DashboardIcon from '@mui/icons-material/Dashboard';
import { useHotkeys } from 'react-hotkeys-hook';

import {
  buildModeSelector,
  darkModeSelector,
  toggleDarkMode,
  toggleBuildMode,
} from './studioSlice';

function Layout({ children }) {
  const [anchorEl, setAnchorEl] = useState(null);
  const dispatch = useDispatch();
  const buildMode = useSelector(buildModeSelector);
  const darkMode = useSelector(darkModeSelector);

  useHotkeys('cmd+b', () => {
    dispatch(toggleBuildMode());
  });

  const theme = useMemo(
    () =>
      createTheme({
        palette: {
          ...(!darkMode
            ? {
                primary: {
                  main: '#212121',
                  light: '#484848',
                  dark: '#000000',
                },
                secondary: colors.deepOrange,
                background: {
                  default: '#fafafa',
                },
              }
            : {
                mode: 'dark',
              }),
        },
        components: {
          MuiSwitch: {
            styleOverrides: {
              track: {
                backgroundColor: '#fff',
              },
            },
          },
        },
      }),
    [darkMode]
  );

  const handleSettingsOpen = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleSettingsClose = () => {
    setAnchorEl(null);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ display: 'flex' }}>
        <MuiAppBar position="fixed" elevation={2}>
          <Toolbar>
            <Grid justifyContent="space-between" alignItems="center" container>
              <Grid item>
                <Button
                  startIcon={<DashboardIcon />}
                  aria-controls="menu-appbar"
                  aria-haspopup="true"
                  color="inherit"
                  // onClick={handleSettingsOpen}
                  variant="text"
                >
                  Menu
                </Button>
                {/* <Menu
                  id="menu-appbar"
                  anchorEl={anchorEl}
                  anchorOrigin={{
                    vertical: 'top',
                    horizontal: 'right',
                  }}
                  keepMounted
                  transformOrigin={{
                    vertical: 'top',
                    horizontal: 'right',
                  }}
                  open={Boolean(anchorEl)}
                  onClose={handleSettingsClose}
                >
                  <MenuItem onClick={handleSettingsClose}>Dataset</MenuItem>
                  <MenuItem onClick={handleSettingsClose}>Settings</MenuItem>
                </Menu> */}
              </Grid>
              <Grid item>
                <Grid container direction="row" alignItems="center">
                  <BubbleChartIcon sx={{ marginRight: 1 }} />
                  <Typography variant="h6" component="div">
                    REPSYS
                  </Typography>
                </Grid>
              </Grid>
              <Grid item>
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
                {/* <FormControlLabel
                  sx={{
                    marginLeft: 1,
                  }}
                  control={
                    <Switch checked={darkMode} onChange={() => dispatch(toggleDarkMode())} />
                  }
                  label="Dark mode"
                /> */}
              </Grid>
            </Grid>
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
