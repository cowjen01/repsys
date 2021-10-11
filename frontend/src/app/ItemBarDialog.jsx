import React from 'react';
import pt from 'prop-types';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import { Formik, Field } from 'formik';

import TextField from './TextField';
import SelectField from './SelectField';

function ItemBarDialog({ open, onClose, initialValues, onSubmit, models }) {
  return (
    <Dialog open={open} fullWidth maxWidth="sm" onClose={onClose}>
      <Formik
        initialValues={initialValues}
        validate={(values) => {
          const errors = {};
          const requiredMessage = 'This field is required';
          if (!values.title) {
            errors.title = requiredMessage;
          }
          return errors;
        }}
        onSubmit={(values, { setSubmitting }) => {
          onSubmit(values);
          setSubmitting(false);
        }}
      >
        {({ submitForm, isSubmitting, values }) => (
          <>
            <DialogContent>
              <Field name="title" label="Title" fullWidth component={TextField} />
              <Field
                name="model"
                fullWidth
                label="Recommendation model"
                component={SelectField}
                options={models.map((m) => ({ label: m.key, value: m.key }))}
              />
              {models
                .find((m) => m.key === values.model)
                .attributes.map((a) => (
                  <Field
                    key={a.key}
                    name={`modelAttributes.${values.model}.${a.key}`}
                    type={a.type || 'text'}
                    fullWidth
                    label={a.label}
                    component={TextField}
                  />
                ))}
              <Field
                name="itemsPerPage"
                fullWidth
                label="Items per page"
                component={SelectField}
                options={[1, 2, 3, 4].map((i) => ({ label: i, value: i }))}
              />
            </DialogContent>
            <DialogActions>
              <Button onClick={onClose}>Close</Button>
              <Button onClick={submitForm} disabled={isSubmitting} autoFocus>
                Submit
              </Button>
            </DialogActions>
          </>
        )}
      </Formik>
    </Dialog>
  );
}

ItemBarDialog.defaultProps = {
  initialValues: {},
  models: [],
};

ItemBarDialog.propTypes = {
  open: pt.bool.isRequired,
  onClose: pt.func.isRequired,
  onSubmit: pt.func.isRequired,
  // eslint-disable-next-line react/forbid-prop-types
  initialValues: pt.any,
  // eslint-disable-next-line react/forbid-prop-types
  models: pt.any,
};

export default ItemBarDialog;
