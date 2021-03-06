/* eslint-disable react/prop-types */
import React from 'react';
import MuiTextField from '@mui/material/TextField';

function TextField({ field: { name, value, onChange, onBlur }, form: { errors }, ...props }) {
  return (
    <MuiTextField
      name={name}
      variant="outlined"
      onChange={onChange}
      onBlur={onBlur}
      fullWidth
      value={value}
      margin="normal"
      helperText={errors[name]}
      error={Boolean(errors[name])}
      {...props}
    />
  );
}

export default TextField;
