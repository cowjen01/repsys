import React from 'react';
import pt from 'prop-types';
import { Alert, AlertTitle } from '@mui/material';

function ErrorAlert({ error }) {
  return (
    <Alert severity="error">
      <AlertTitle>API Error ({error.status})</AlertTitle>
      {error.data.message || JSON.stringify(error.data)}
    </Alert>
  );
}

ErrorAlert.propTypes = {
  // eslint-disable-next-line react/forbid-prop-types
  error: pt.any.isRequired,
};

export default ErrorAlert;
