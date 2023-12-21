const { exec } = require("node:child_process");
// run the `ls` command using exec
exec("python test.py 1 85 66 29 0 26.6 0.351 31", (err, output) => {
  // once the command has completed, the callback function is called
  if (err) {
    // log and return if we encounter an error
    console.error("could not execute command: ", err);
    return;
  }
  // log the output received from the command
  console.log("Output: \n", output);
});
