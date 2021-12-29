import React, { useState } from 'react';
import pt from 'prop-types';
import {
  Box,
  Toolbar,
  AppBar,
  Typography,
  Switch,
  FormControlLabel,
  Button,
  MenuItem,
  Menu,
  Grid,
  ListItemText,
  ListItemIcon,
  Divider,
} from '@mui/material';
import { useDispatch, useSelector } from 'react-redux';
import { useHotkeys } from 'react-hotkeys-hook';
import BubbleChartIcon from '@mui/icons-material/BubbleChart';
import DashboardIcon from '@mui/icons-material/Dashboard';
// import DonutSmallIcon from '@mui/icons-material/DonutSmall';
import TableRowsIcon from '@mui/icons-material/TableRows';
import SettingsIcon from '@mui/icons-material/Settings';

import { openSettingsDialog } from '../reducers/dialogs';
import { buildModeSelector, toggleBuildMode } from '../reducers/root';

function Layout({ children }) {
  const dispatch = useDispatch();
  const buildMode = useSelector(buildModeSelector);

  const [menuAnchor, setMenuAnchor] = useState(null);

  useHotkeys('cmd+b', () => {
    dispatch(toggleBuildMode());
  });

  const handleMenuOpen = (event) => {
    setMenuAnchor(event.currentTarget);
  };

  const handleMenuClose = () => {
    setMenuAnchor(null);
  };

  return (
    <>
      <AppBar position="fixed" elevation={2}>
        <Toolbar>
          <Grid justifyContent="space-between" alignItems="center" container>
            <Grid item>
              <Button
                startIcon={<DashboardIcon />}
                color="inherit"
                onClick={handleMenuOpen}
                variant="text"
              >
                Menu
              </Button>
              <Menu
                anchorEl={menuAnchor}
                anchorOrigin={{
                  vertical: 'top',
                  horizontal: 'right',
                }}
                keepMounted
                transformOrigin={{
                  vertical: 'top',
                  horizontal: 'right',
                }}
                open={Boolean(menuAnchor)}
                onClose={handleMenuClose}
              >
                <MenuItem onClick={handleMenuClose} selected>
                  <ListItemIcon>
                    <TableRowsIcon fontSize="small" />
                  </ListItemIcon>
                  <ListItemText>Recommenders</ListItemText>
                </MenuItem>
                {/* <MenuItem onClick={handleMenuClose}>
                  <ListItemIcon>
                    <DonutSmallIcon fontSize="small" />
                  </ListItemIcon>
                  <ListItemText>Dataset Analysis</ListItemText>
                </MenuItem> */}
                <Divider />
                <MenuItem
                  onClick={() => {
                    dispatch(openSettingsDialog());
                    handleMenuClose();
                  }}
                >
                  <ListItemIcon>
                    <SettingsIcon fontSize="small" />
                  </ListItemIcon>
                  <ListItemText>Settings</ListItemText>
                </MenuItem>
              </Menu>
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
                    sx={{
                      '& .MuiSwitch-track': {
                        backgroundColor: '#fff',
                      },
                    }}
                    onChange={() => dispatch(toggleBuildMode())}
                  />
                }
                // label={buildMode ? 'Build Mode' : 'Preview Mode'}
                label="Build Mode"
              />
            </Grid>
          </Grid>
        </Toolbar>
      </AppBar>
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          height: '100vh',
          overflow: 'auto',
          paddingTop: 3,
          paddingBottom: 4,
        }}
      >
        <Toolbar />
        {children}
      </Box>
    </>
  );
}

Layout.propTypes = {
  children: pt.node.isRequired,
};

export default Layout;
