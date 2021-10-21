/* eslint-disable react/prop-types */
import React from 'react';
import MuiTextField from '@mui/material/TextField';

function TextField({
  field: { name, value, onChange, onBlur },
  form: { errors, touched },
  ...props
}) {
  return (
    <MuiTextField
      name={name}
      variant="outlined"
      onChange={onChange}
      onBlur={onBlur}
      value={value}
      margin="normal"
      helperText={touched[name] && errors[name]}
      error={Boolean(touched[name] && errors[name])}
      {...props}
    />
  );
}

export default TextField;
