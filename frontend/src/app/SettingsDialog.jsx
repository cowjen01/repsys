import React from 'react';
import pt from 'prop-types';
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

function SettingsDialog({ open, onClose }) {
  const darkMode = useSelector(darkModeSelector);
  const dispatch = useDispatch();

  return (
    <Dialog open={open} fullWidth maxWidth="sm" onClose={() => onClose(false)}>
      <DialogTitle>Repsys Settings</DialogTitle>
      <DialogContent>
        <FormGroup>
          <FormControlLabel
            control={<Switch checked={darkMode} onChange={() => dispatch(toggleDarkMode())} />}
            label="Dark Mode"
          />
        </FormGroup>
      </DialogContent>
      <DialogActions>
        <Button color="secondary" onClick={() => onClose(false)}>
          Close
        </Button>
      </DialogActions>
    </Dialog>
  );
}

SettingsDialog.propTypes = {
  open: pt.bool.isRequired,
  onClose: pt.func.isRequired,
};

export default SettingsDialog;
