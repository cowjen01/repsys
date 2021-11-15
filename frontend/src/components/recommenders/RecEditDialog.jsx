import React, { useMemo } from 'react';
import pt from 'prop-types';
import { Formik, Field } from 'formik';
import { Button, Dialog, DialogActions, DialogContent, DialogTitle } from '@mui/material';
import { useDispatch, useSelector } from 'react-redux';

import { addRecommender, updateRecommender } from '../../reducers/recommenders';
import { recEditDialogSelector, closeRecEditDialog, openSnackbar } from '../../reducers/dialogs';
import { TextField, SelectField, CheckboxField } from '../fields';

function RecEditDialog({ models }) {
  const dialog = useSelector(recEditDialogSelector);
  const dispatch = useDispatch();

  const handleClose = () => {
    dispatch(closeRecEditDialog());
  };

  const handleSubmit = (values) => {
    const data = {
      ...values,
      itemsPerPage: parseInt(values.itemsPerPage, 10),
    };
    if (!values.id) {
      dispatch(addRecommender(data));
    } else {
      dispatch(updateRecommender(data));
    }
    dispatch(
      openSnackbar({
        message: 'All settings applied!',
      })
    );
    handleClose();
  };

  return (
    <Dialog open={dialog.open} fullWidth maxWidth="sm" onClose={handleClose}>
      <DialogTitle>Recommender Configuration</DialogTitle>
      <Formik
        initialValues={dialog.data}
        validate={(values) => {
          const errors = {};
          const requiredMessage = 'This field is required';
          if (!values.title) {
            errors.title = requiredMessage;
          }
          if (!values.itemsLimit) {
            errors.itemsLimit = requiredMessage;
          }
          return errors;
        }}
        onSubmit={(values, { setSubmitting }) => {
          handleSubmit(values);
          setSubmitting(false);
        }}
      >
        {({ submitForm, isSubmitting, values }) => {
          const model = useMemo(() => models.find((m) => m.key === values.model), [values.model]);
          return (
            <>
              <DialogContent>
                <Field name="title" label="Title" fullWidth component={TextField} />
                <Field
                  name="itemsPerPage"
                  label="Items per page"
                  component={SelectField}
                  options={[1, 2, 3, 4].map((i) => ({ label: i, value: i }))}
                />
                <Field
                  name="itemsLimit"
                  label="Max number of items"
                  component={TextField}
                  type="number"
                />
                <Field
                  name="model"
                  label="Recommendation model"
                  component={SelectField}
                  options={[...models.map((m) => ({ label: m.key, value: m.key }))]}
                />
                {model &&
                  model.params &&
                  model.params.map((a) => {
                    const name = `modelParams.${values.model}.${a.key}`;
                    if (a.type === 'select') {
                      return (
                        <Field
                          key={a.key}
                          name={name}
                          label={a.label}
                          component={SelectField}
                          options={a.options.map((b) => ({ label: b, value: b }))}
                        />
                      );
                    }
                    if (a.type === 'bool') {
                      return (
                        <Field key={a.key} name={name} label={a.label} component={CheckboxField} />
                      );
                    }
                    return (
                      <Field
                        key={a.key}
                        name={name}
                        label={a.label}
                        component={TextField}
                        type={a.type}
                      />
                    );
                  })}
              </DialogContent>
              <DialogActions>
                <Button onClick={handleClose} color="secondary">
                  Close
                </Button>
                <Button onClick={submitForm} color="secondary" disabled={isSubmitting} autoFocus>
                  Save
                </Button>
              </DialogActions>
            </>
          );
        }}
      </Formik>
    </Dialog>
  );
}

RecEditDialog.defaultProps = {
  models: [],
};

RecEditDialog.propTypes = {
  // eslint-disable-next-line react/forbid-prop-types
  models: pt.any,
};

export default RecEditDialog;
