using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;

namespace WebApplication1.Models
{
    public class Car
    {
        public int id { get; set; }
        public double leftspeed { get; set; }
        public double rightspeed { get;set; }

        public bool leftlinesensor { get; set; }

        public bool rightlinesensor { get; set; }

        public double ultrasonicsensor { get; set; }
    }
}