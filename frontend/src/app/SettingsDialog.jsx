import React from 'react';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import FormGroup from '@mui/material/FormGroup';
import FormControlLabel from '@mui/material/FormControlLabel';
import Switch from '@mui/material/Switch';
import DialogTitle from '@mui/material/DialogTitle';
import { useDispatch, useSelector } from 'react-redux';

import { darkModeSelector, toggleDarkMode } from '../reducers/settings';
import { closeSettingsDialog, settingsDialogOpenSelector } from '../reducers/studio';

function SettingsDialog() {
  const darkMode = useSelector(darkModeSelector);
  const dialogOpen = useSelector(settingsDialogOpenSelector);
  const dispatch = useDispatch();

  const handleClose = () => {
    dispatch(closeSettingsDialog());
  };

  return (
    <Dialog open={dialogOpen} fullWidth maxWidth="sm" onClose={handleClose}>
      <DialogTitle>Repsys Settings</DialogTitle>
      <DialogContent>
        <FormGroup>
          <FormControlLabel
            control={
              <Switch
                color="primary"
                checked={darkMode}
                onChange={() => dispatch(toggleDarkMode())}
              />
            }
            label="Dark Mode"
          />
        </FormGroup>
      </DialogContent>
      <DialogActions>
        <Button color="secondary" onClick={handleClose}>
          Close
        </Button>
      </DialogActions>
    </Dialog>
  );
}

export default SettingsDialog;
