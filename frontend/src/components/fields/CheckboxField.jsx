/* eslint-disable react/prop-types */
import React from 'react';
import Checkbox from '@mui/material/Checkbox';
import FormControlLabel from '@mui/material/FormControlLabel';

function CheckboxField({ field: { name, value, onChange }, label, ...props }) {
  return (
    <FormControlLabel
      sx={{ width: '100%' }}
      control={<Checkbox name={name} onChange={onChange} checked={value} {...props} />}
      label={label}
    />
  );
}

export default CheckboxField;
