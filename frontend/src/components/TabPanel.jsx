import React from 'react';
import pt from 'prop-types';

function TabPanel({ children, value, index, padding, ...other }) {
  return (
    <div role="tabpanel" hidden={value !== index} {...other}>
      {value === index ? children : null}
    </div>
  );
}

TabPanel.defaultProps = {
  padding: 0,
};

TabPanel.propTypes = {
  children: pt.element.isRequired,
  value: pt.number.isRequired,
  index: pt.number.isRequired,
  padding: pt.number,
};

export default TabPanel;
